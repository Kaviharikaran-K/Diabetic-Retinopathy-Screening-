function telemedicine_sim()
% telemedicine_sim.m
% ======================================================================
% Down-stream Link -- MATLAB SimEvents discrete-event model of a rural
% telemedicine DR screening network fed by the Python XAI pipeline.
%
% The Python engine (pipeline_engine.py) exports:
%     exports/batch_sim_metrics.json      machine-readable latency + triage
%     exports/risk_priority_queue.csv     per-eye risk/priority rows
%
% This script:
%   1. Loads those Python-generated metrics (latencies, priorities).
%   2. Builds a SimEvents queueing model for a district serving >100,000
%      patients / year across three nodes:
%        Patient arrivals  ->  AI screening server (Python engine)
%                           ->  Bandwidth / uplink transfer (rural link)
%                           ->  Risk-priority specialist review queue
%   3. Runs the discrete-event simulation.
%   4. Also computes an independent analytic M/M/1 & M/M/c estimate AND a
%      lightweight pure-MATLAB discrete-event mirror so results remain
%      meaningful even if a specific SimEvents library name differs on the
%      user's MATLAB release.
%
% Usage:
%     >> telemedicine_sim          % reads Python exports if present
%
% Dependencies: Simulink + SimEvents (optional -- graceful degradation).
% No toolbox-specific random functions are used: exponential draws are
% computed as -log(rand)/mu.
% ======================================================================

% ----------------------------------------------------------------------
% 1. User-adjustable scenario parameters (district scale).
% ----------------------------------------------------------------------
cfg = struct();
cfg.annualPatients    = 105000;          % district 100k+ patients / year
cfg.clinicHoursPerDay = 10;               % operating hours of the clinic
cfg.clinicDaysPerYear = 300;              % working days (6 days/week, 50 wks)
cfg.simClinicDays     = 30;               % length of the simulated period
cfg.aiServers         = 1;                % AI screening stations
cfg.specialists       = 3;                % reviewing ophthalmologists / graders
cfg.uploadKbps        = 256;              % rural uplink bandwidth
cfg.imageSizeKb       = 2048;             % per-eye scan (JPEG about 2 MB)
cfg.retransmission    = 1.06;             % TCP overhead / retries factor
cfg.specialistServiceMeanSec = 150;       % mean expert review duration
cfg.seed              = 42;
cfg.metricsJson       = 'exports/batch_sim_metrics.json';
cfg.metricsCsv        = 'exports/risk_priority_queue.csv';
cfg.reportCsv         = 'telemedicine_sim_results.csv';

% ----------------------------------------------------------------------
% 2. Load Python-generated metrics (latencies + priority distribution).
% ----------------------------------------------------------------------
m = loadPythonMetrics(cfg.metricsJson, cfg.metricsCsv);
if ~isempty(m)
    fprintf('[telemedicine_sim] Loaded Python metrics from %s\n', m.source);
end

% Derived Poisson arrival intensity (patients / second, clinic hours only).
arrivalsPerDay = cfg.annualPatients / cfg.clinicDaysPerYear;
lambda         = arrivalsPerDay / (cfg.clinicHoursPerDay * 3600);

% ---- service-time means (seconds) ------------------------------------
% AI stage : per-case latency of the XAI pipeline (both eyes) + handshake.
if ~isempty(m)
    aiMeanT = m.latencyTotalSec + 0.35;   % pipeline + I/O overhead
else
    aiMeanT = 0.75;                       % calibrated default
end
aiMeanT = max(0.05, aiMeanT / cfg.aiServers);

% Bandwidth stage: two eyes * image size / effective uplink, with overhead.
bwMeanT = (2 * cfg.imageSizeKb * 1024 * 8) / (cfg.uploadKbps * 1000) ...
          * cfg.retransmission;

% Specialist stage (exponential service, M/M/c style).
specMeanT = cfg.specialistServiceMeanSec;

% ---- priority distribution from the Python triage file ------------------
if ~isempty(m) && isfield(m, 'priorityWeights')
    wts = m.priorityWeights;            % [P0 P1 P2 P3] frequencies
else
    wts = [0.05 0.15 0.30 0.50];        % fallback district mix
end
wts = wts(:)' / sum(wts);               % normalise
priMean = sum((0:3) .* wts);
priStd  = sqrt(sum(((0:3) - priMean).^2 .* wts));

% ----------------------------------------------------------------------
% 3. Independent estimates -- analytic queueing theory + code-based DES.
% ----------------------------------------------------------------------
an  = analyticEstimates(lambda, aiMeanT, bwMeanT, specMeanT, ...
                        cfg.aiServers, cfg.specialists);
des = desRun(struct('lambda',  lambda, ...
                    'muAi',    1 / aiMeanT, ...
                    'muBw',    1 / bwMeanT, ...
                    'muSpec',  1 / specMeanT, ...
                    'nSpec',   cfg.specialists, ...
                    'maxT',    cfg.simClinicDays * cfg.clinicHoursPerDay * 3600, ...
                    'wts',     wts, ...
                    'seed',    cfg.seed));

% ----------------------------------------------------------------------
% 4. SimEvents model (graceful degradation if block names differ).
% ----------------------------------------------------------------------
simResult = buildAndRunSimEvents(cfg, lambda, aiMeanT, bwMeanT, ...
                                 specMeanT, priMean, priStd);

% ----------------------------------------------------------------------
% 5. Consolidated report.
% ----------------------------------------------------------------------
report = summarize(cfg, arrivalsPerDay, lambda, aiMeanT, bwMeanT, ...
                   specMeanT, an, des, simResult);
writeReport(cfg.reportCsv, report);
disp(report.text);

end % telemedicine_sim


% ======================================================================
% Local helpers ---------------------------------------------------------
% ======================================================================

% --- Metrics loading --------------------------------------------------
function m = loadPythonMetrics(jsonFile, csvFile)
m = [];
% JSON path (R2016b+ has jsondecode)
try
    if isfile(jsonFile)
        raw = jsondecode(fileread(jsonFile));
        op = raw.operator_latency_ms;           % struct of ms values
        m.latencyTotalSec = op.total_ms / 1000;
        q = raw.risk_priority_queue;            % array of per-eye dicts
        p = arrayfun(@(r) r.priority, q);       % priority numbers
        m.priorityWeights = zeros(1, 4);
        m.priorityWeights(1) = mean(p == 0);
        m.priorityWeights(2) = mean(p == 1);
        m.priorityWeights(3) = mean(p == 2);
        m.priorityWeights(4) = mean(p == 3);
        m.source = jsonFile;
        return;
    end
catch err %#ok<NASGU>
end
% CSV fallback
try
    if isfile(csvFile)
        t = readtable(csvFile, 'PreserveVariableNames', true);
        if any(strcmp(t.Properties.VariableNames, 'priority'))
            pr = t.priority;
            m.priorityWeights = zeros(1, 4);
            m.priorityWeights(1) = mean(pr == 0);
            m.priorityWeights(2) = mean(pr == 1);
            m.priorityWeights(3) = mean(pr == 2);
            m.priorityWeights(4) = mean(pr == 3);
            m.latencyTotalSec = 0.5;    % CSV carries no latency fields
            m.source = csvFile;
            return;
        end
    end
catch err %#ok<NASGU>
end
end

% --- Analytic M/M/1 and M/M/c estimates --------------------------------
function an = analyticEstimates(lambda, aiMean, bwMean, specMean, nAi, nSpec)
rhoAi = lambda * aiMean / nAi;
WqAi  = mm1_wait(lambda, aiMean, nAi);
rhoBw = lambda * bwMean;
WqBw  = mm1_wait(lambda, bwMean, 1);
rhoSp = lambda * specMean / nSpec;
WqSp  = mm1_wait(lambda, specMean, nSpec);

an.lambda = lambda;
an.utilAi = rhoAi;
an.utilBw = rhoBw;
an.utilSp = rhoSp;
an.WqAi   = WqAi;
an.WqBw   = WqBw;
an.WqSp   = WqSp;
an.WTroTotal = WqAi + aiMean + WqBw + bwMean + WqSp + specMean;
end

function Wq = mm1_wait(lambda, meanService, servers)
% M/M/c waiting-time estimate (Erlang-C for c > 1).
mu  = 1 / meanService;
rho = lambda * meanService / servers;
if rho >= 0.9999
    Wq = Inf;
    return;
end
if servers == 1
    Wq = (rho * meanService) / (1 - rho);
else
    % Erlang-C: probability an arrival has to wait
    c = servers;
    sumTerm = 0;
    for k = 0:c-1
        sumTerm = sumTerm + (c * rho)^k / factorial(k);
    end
    p0 = 1 / (sumTerm + (c * rho)^c / (factorial(c) * (1 - rho)));
    pd = (c * rho)^c / (factorial(c) * (1 - rho)) * p0;
    Wq = pd * meanService / (c - lambda * meanService);
end
end

% --- Pure-MATLAB discrete-event mirror (risk-priority queue) ------------
function R = desRun(p)
% Event-driven simulation of: arrivals -> AI -> bandwidth -> specialists.
% Specialist queue is priority-ordered (P0 rt. through P3, FIFO within class).
R = struct('nArrived', 0, 'nCompleted', 0, ...
           'waitAi', 0, 'waitBw', 0, 'waitSpec', 0, 'respSum', 0, ...
           'utilAi', 0, 'utilBw', 0, 'utilSp', 0, ...
           'avgQa', 0, 'avgQb', 0, 'avgQs', 0, ...
           'maxQa', 0, 'maxQb', 0, 'maxQs', 0, 'simTime', 0, 'avgResp', 0);
rng(p.seed);

% ---- event set: [tt, type, serverIndex] --------------------------------
evT = [];  evTy = strings(0, 1);  evIdx = [];

% ---- queues (FIFO for AI/BW, priority for specialists) ------------------
qAt = []; qA0 = []; qAp = [];      % AI queue  : enqueue time / origin time / priority
qBt = []; qB0 = []; qBp = [];      % BW queue
qSt = []; qS0 = []; qSp = [];      % specialist queue (priority ordered on dequeue)

% ---- server state --------------------------------------------------------
aiEnd = inf; aiA0 = 0; aiPr = 0;   % end-of-service time, origin time, priority
bwEnd = inf; bwA0 = 0; bwPr = 0;
spEnd = inf(1, p.nSpec); spA0 = inf(1, p.nSpec);

% ---- statistics accumulators ----------------------------------------------
nArr = 0; nDone = 0;
waitAi = 0; waitBw = 0; waitSp = 0; respSum = 0;
busyAi = 0; busyBw = 0; busySp = 0;
intQa = 0; intQb = 0; intQs = 0;
maxQa = 0; maxQb = 0; maxQs = 0;
t = 0; prev = 0;

    % ---- nested event handlers (share the enclosing workspace) ------------
    function s = expS(mu)
        s = -log(rand) / max(mu, 1e-12);
    end

    function pr = drawPriority()
        u = rand();
        c = cumsum(p.wts);
        idx = find(u <= c, 1, 'first') - 1;
        if isempty(idx), idx = 3; end
        pr = idx;
    end

    function onArrival()
        nArr = nArr + 1;
        pr = drawPriority();
        a0 = t;
        if isinf(aiEnd)
            aiEnd = t + expS(p.muAi); aiA0 = a0; aiPr = pr;
            evT(end + 1, 1) = aiEnd; evTy(end + 1, 1) = "aiD"; evIdx(end + 1, 1) = 0; %#ok<AGROW>
        else
            qAt(end + 1) = t; qA0(end + 1) = a0; qAp(end + 1) = pr; %#ok<AGROW>
        end
        evT(end + 1, 1) = t + expS(p.lambda); evTy(end + 1, 1) = "arr"; evIdx(end + 1, 1) = 0; %#ok<AGROW>
    end

    function onAiDone()
        doneA0 = aiA0; donePr = aiPr;      % entity leaving the AI server
        aiEnd = inf;
        if ~isempty(qAt)
            waitAi = waitAi + (t - qAt(1));
            nxtA0 = qA0(1); nxtPr = qAp(1);
            qAt(1) = []; qA0(1) = []; qAp(1) = [];
            aiEnd = t + expS(p.muAi); aiA0 = nxtA0; aiPr = nxtPr;
            evT(end + 1, 1) = aiEnd; evTy(end + 1, 1) = "aiD"; evIdx(end + 1, 1) = 0; %#ok<AGROW>
        end
        handToBw(doneA0, donePr);
    end

    function handToBw(a0, pr)
        if isinf(bwEnd)
            bwEnd = t + expS(p.muBw); bwA0 = a0; bwPr = pr;
            evT(end + 1, 1) = bwEnd; evTy(end + 1, 1) = "bwD"; evIdx(end + 1, 1) = 0; %#ok<AGROW>
        else
            qBt(end + 1) = t; qB0(end + 1) = a0; qBp(end + 1) = pr; %#ok<AGROW>
        end
    end

    function onBwDone()
        doneA0 = bwA0; donePr = bwPr;      % entity leaving the bandwidth server
        bwEnd = inf;
        if ~isempty(qBt)
            waitBw = waitBw + (t - qBt(1));
            nxtA0 = qB0(1); nxtPr = qBp(1);
            qBt(1) = []; qB0(1) = []; qBp(1) = [];
            bwEnd = t + expS(p.muBw); bwA0 = nxtA0; bwPr = nxtPr;
            evT(end + 1, 1) = bwEnd; evTy(end + 1, 1) = "bwD"; evIdx(end + 1, 1) = 0; %#ok<AGROW>
        end
        handToSpec(doneA0, donePr);
    end

    function handToSpec(a0, pr)
        free = find(spEnd == inf, 1);
        if isempty(free)
            qSt(end + 1) = t; qS0(end + 1) = a0; qSp(end + 1) = pr; %#ok<AGROW>
            return;
        end
        startSpec(free, a0);
    end

    function startSpec(i, a0)
        spEnd(i) = t + expS(p.muSpec); spA0(i) = a0;
        evT(end + 1, 1) = spEnd(i); evTy(end + 1, 1) = "spD"; evIdx(end + 1, 1) = i; %#ok<AGROW>
    end

    function finishSpec(i)
        respSum = respSum + (t - spA0(i));   % end-to-end latency
        nDone   = nDone + 1;
        spEnd(i) = inf; spA0(i) = inf;
        if ~isempty(qSt)
            [~, bidx] = min(qSp);            % highest priority (lowest number) first
            waitSp = waitSp + (t - qSt(bidx));
            a0 = qS0(bidx);
            qSt(bidx) = []; qS0(bidx) = []; qSp(bidx) = [];
            startSpec(i, a0);
        end
    end

% ---- seed the first arrival -----------------------------------------------
r0 = -log(rand) / max(p.lambda, 1e-12);
evT(1) = r0; evTy(1) = "arr"; evIdx(1) = 0;

% ---- main event loop --------------------------------------------------------
while ~isempty(evT)
    [t, mi] = min(evT);
    ty = evTy(mi); ii = evIdx(mi);
    evT(mi) = []; evTy(mi) = []; evIdx(mi) = [];

    if t > p.maxT
        break;
    end
    dt = t - prev; prev = t;

    if isfinite(aiEnd), busyAi = busyAi + dt; end
    if isfinite(bwEnd), busyBw = busyBw + dt; end
    busySp = busySp + sum(isfinite(spEnd)) * dt;

    nqA = numel(qAt); nqB = numel(qBt); nqS = numel(qSt);
    intQa = intQa + nqA * dt;
    intQb = intQb + nqB * dt;
    intQs = intQs + nqS * dt;
    maxQa = max(maxQa, nqA);
    maxQb = max(maxQb, nqB);
    maxQs = max(maxQs, nqS);

    switch ty
        case "arr",  onArrival();
        case "aiD",  onAiDone();
        case "bwD",  onBwDone();
        case "spD",  finishSpec(ii);
        otherwise,   error('desRun: unknown event type "%s"', ty);
    end
end

% ---- wrap-up statistics -------------------------------------------------------
simT = prev;
if simT <= 0, simT = p.maxT; end
R.nArrived = nArr;
R.nCompleted = nDone;
R.waitAi = waitAi; R.waitBw = waitBw; R.waitSpec = waitSp;
R.respSum = respSum;
R.utilAi = busyAi / simT;
R.utilBw = busyBw / simT;
R.utilSp = busySp / (simT * p.nSpec);
R.avgQa = intQa / simT; R.avgQb = intQb / simT; R.avgQs = intQs / simT;
R.maxQa = maxQa; R.maxQb = maxQb; R.maxQs = maxQs;
R.simTime = simT;
R.avgResp = respSum / max(nDone, 1);
end

% --- SimEvents model construction & execution -------------------------------
function result = buildAndRunSimEvents(cfg, lambda, aiMean, bwMean, ...
                                       specMean, priMean, priStd)
result = struct('built', false, 'ran', false, 'reason', 'not attempted', ...
                'queueProbe', false, 'avgSpecQueue', NaN, 'maxSpecQueue', NaN);
model = 'DRTelemedicineSim';

% ---- availability check -----------------------------------------------------
try
    load_system('simeventslib');        % modern releases
catch err %#ok<NASGU>
    try
        load_system('simevents');       % legacy SimEvents library name
    catch err2 %#ok<NASGU>
        fprintf(['[telemedicine_sim] SimEvents library not found -- using ', ...
                 'analytic + DES estimates only.\n']);
        result.reason = 'SimEvents not installed';
        return;
    end
end

if bdIsLoaded(model)
    close_system(model, 0);
end

try
    new_system(model);
    try
        save_system(model);     % best-effort persist (needed on some releases)
    catch err %#ok<NASGU>
    end
    set_param(model, 'StopTime', ...
        sprintf('%.1f', cfg.simClinicDays * cfg.clinicHoursPerDay * 3600));

    % ---- blocks ------------------------------------------------------------
    gen   = addBlockTolerant(model, 'Patient_Generator', ...
        {'simeventslib/Time-Based Entity Generator', ...
         'simeventslib/Generators/Time-Based Entity Generator', ...
         'simeventslib/Entity Generator'});
    aiQ   = addBlockTolerant(model, 'AI_Queue', {'simeventslib/Entity Queue'});
    aiSrv = addBlockTolerant(model, 'AI_Server', {'simeventslib/Entity Server'});
    bwSrv = addBlockTolerant(model, 'Bandwidth_Server', {'simeventslib/Entity Server'});
    specQ = addBlockTolerant(model, 'Specialist_Queue', ...
        {'simeventslib/Entity Priority Queue', ...
         'simeventslib/Queues/Entity Priority Queue', ...
         'simeventslib/Entity Queue'});
    specSrv = addBlockTolerant(model, 'Specialist_Server', {'simeventslib/Entity Server'});
    term = addBlockTolerant(model, 'Terminator', ...
        {'simeventslib/Entity Terminator', ...
         'simeventslib/Terminators/Entity Terminator'});

    % ---- parameters -----------------------------------------------------------
    interExpr = sprintf('-%g*log(rand)', 1 / max(lambda, 1e-12));
    setParamTolerant([model '/Patient_Generator'], ...
        {'IntergenerationTime', 'InterGenerationTime'}, interExpr);
    setParamTolerant([model '/Patient_Generator'], 'InitialEntities', '0');

    aiExpr = sprintf('-%g*log(rand)', aiMean);
    bwExpr = sprintf('-%g*log(rand)', bwMean);
    spExpr = sprintf('-%g*log(rand)', specMean);
    setParamTolerant([model '/AI_Server'], 'ServiceTime', aiExpr);
    setParamTolerant([model '/Bandwidth_Server'], 'ServiceTime', bwExpr);
    setParamTolerant([model '/Specialist_Server'], 'ServiceTime', spExpr);

    % ---- risk-priority attribute (best effort across releases) ----------------
    priExpr = sprintf('max(0,min(3,round(%g+%g*randn)))', priMean, max(priStd, 0.5));
    try
        set_param([model '/Patient_Generator'], 'AttributeNames', "{'priority'}");
        set_param([model '/Patient_Generator'], 'AttributeValues', "{'" priExpr "'}");
        okPrio = setParamTolerant([model '/Specialist_Queue'], ...
            {'Priority', 'PriorityName', 'PriorityAttribute'}, 'priority');
        if okPrio
            fprintf('[telemedicine_sim] Risk-priority queue active (P0..P3).\n');
        end
    catch err %#ok<NASGU>
        fprintf(['[telemedicine_sim] Priority attribute unavailable -- ', ...
                 'specialist queue runs FIFO.\n']);
    end

    % ---- wiring ----------------------------------------------------------------
    add_line(model, [gen '/1'],    [aiQ '/1']);
    add_line(model, [aiQ '/1'],    [aiSrv '/1']);
    add_line(model, [aiSrv '/1'],  [bwSrv '/1']);
    add_line(model, [bwSrv '/1'],  [specQ '/1']);
    add_line(model, [specQ '/1'],  [specSrv '/1']);
    add_line(model, [specSrv '/1'], [term '/1']);

    % ---- optional live statistics probe on the specialist queue ----------------
    probeVar = 'simSpecQueueLen';
    probeAdded = false;
    for cand = {'built-in/ToWorkspace', 'simulink/Sinks/To Workspace'}
        try
            add_block(cand{1}, [model '/Specialist_QLen_Probe']);
            set_param([model '/Specialist_QLen_Probe'], 'VariableName', probeVar);
            set_param([model '/Specialist_QLen_Probe'], 'SaveFormat', 'Array');
            add_line(model, [specQ '/2'], [model '/Specialist_QLen_Probe/1']);
            probeAdded = true;
            break;
        catch err %#ok<NASGU>
        end
    end
    result.queueProbe = probeAdded;

    result.built = true;

    % ---- execute ---------------------------------------------------------------
    simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
    result.ran = true;

    try
        ql = simOut.get(probeVar);              % rows = samples
        qcol = double(ql(:, 1));
        result.avgSpecQueue = mean(qcol);
        result.maxSpecQueue = max(qcol);
    catch err %#ok<NASGU>
        % no probe data -- DES/analytic figures cover reporting
    end

    close_system(model, 0);
catch err
    fprintf('[telemedicine_sim] SimEvents model error: %s\n', err.message);
    try
        close_system(model, 0);
    catch
    end
    result.reason = err.message;
end
end

% --- helper: add a block by trying several SimEvents library paths ----------
function added = addBlockTolerant(model, name, candidates)
callerErr = '';
for k = 1:numel(candidates)
    try
        add_block(candidates{k}, [model '/' name]);
        added = [model '/' name];
        return;
    catch err
        callerErr = err.message;
    end
end
error('addBlockTolerant: could not add "%s" to %s (%s). Tried: %s', ...
      name, model, callerErr, strjoin(candidates, ' | '));
end

% --- helper: tolerant set_param over a list of parameter names --------------
function ok = setParamTolerant(blockPath, paramNames, value)
ok = false;
for k = 1:numel(paramNames)
    try
        set_param(blockPath, paramNames{k}, value);
        ok = true;
        return;
    catch err %#ok<NASGU>
    end
end
end

% --- summary report ------------------------------------------------------------
function report = summarize(cfg, arrivalsPerDay, lambda, aiMean, bwMean, ...
                            specMean, an, des, sim)
nDone = max(des.nCompleted, 1);
desWqAi = des.waitAi / nDone;
desWqBw = des.waitBw / nDone;
desWqSp = des.waitSpec / nDone;
desRespMin = des.avgResp / 60;

lines = string([]);
lines(end + 1) = '=================================================================';
lines(end + 1) = ' DR TELEMEDICINE -- DISCRETE-EVENT SCREENING NETWORK REPORT';
lines(end + 1) = '=================================================================';
lines(end + 1) = sprintf(' District scale      : %d patients / year (~%d / day)', ...
    cfg.annualPatients, round(arrivalsPerDay));
lines(end + 1) = sprintf(' Poisson arrival rate: %.5f / sec (mean IAT %.1f s)', ...
    lambda, 1 / lambda);
lines(end + 1) = sprintf(' AI decode + classify: %.2f s mean service (Python engine)', aiMean);
lines(end + 1) = sprintf(' Rural uplink        : %.1f s mean transfer @ %g kbps', ...
    bwMean, cfg.uploadKbps);
lines(end + 1) = sprintf(' Specialist review   : %.1f s mean service, %d specialists', ...
    specMean, cfg.specialists);
lines(end + 1) = '-----------------------------------------------------------------';
lines(end + 1) = sprintf(' %-22s %12s %12s %12s', 'Stage', 'Analytic', 'DES', 'SimEvents');
lines(end + 1) = sprintf(' %-22s %12s %12s %12s', 'AI utilization', ...
    fmtPct(an.utilAi), fmtPct(des.utilAi), '--');
lines(end + 1) = sprintf(' %-22s %12s %12s %12s', 'BW utilization', ...
    fmtPct(an.utilBw), fmtPct(des.utilBw), '--');
lines(end + 1) = sprintf(' %-22s %12s %12s %12s', 'Specialist utilization', ...
    fmtPct(an.utilSp), fmtPct(des.utilSp), '--');
lines(end + 1) = sprintf(' %-22s %12s %12s %12s', 'Wait @ AI (s)', ...
    fmtSec(an.WqAi), fmtSec(desWqAi), '--');
lines(end + 1) = sprintf(' %-22s %12s %12s %12s', 'Wait @ bandwidth (s)', ...
    fmtSec(an.WqBw), fmtSec(desWqBw), '--');
lines(end + 1) = sprintf(' %-22s %12s %12s %12s', 'Wait @ specialist (s)', ...
    fmtSec(an.WqSp), fmtSec(desWqSp), '--');
lines(end + 1) = sprintf(' %-22s %12s %12s %12s', 'End-to-end resp. (min)', ...
    fmtMin(an.WTroTotal), fmtMin(desRespMin), '--');
lines(end + 1) = sprintf(' %-22s %12s %12s %12s', 'Avg specialist queue', ...
    fmtVal(mm1LambdaWq(an.WqSp, lambda)), fmtVal(des.avgQs), fmtVal(sim.avgSpecQueue));
lines(end + 1) = '-----------------------------------------------------------------';
if sim.built && sim.ran
    lines(end + 1) = sprintf([' SimEvents model "DRTelemedicineSim" built and run ', ...
        'over %d clinic days.'], cfg.simClinicDays);
    if sim.queueProbe
        lines(end + 1) = sprintf([' Specialist queue length sampled live ', ...
            '(mean %.2f, peak %.2f).'], sim.avgSpecQueue, sim.maxSpecQueue);
    end
else
    lines(end + 1) = sprintf(' SimEvents model not simulated (%s).', sim.reason);
end
lines(end + 1) = '=================================================================';

report.csvRows = table( ...
    arrivalsPerDay, lambda, aiMean, bwMean, specMean, ...
    an.utilAi, an.utilSp, an.WqAi, an.WqSp, an.WTroTotal, ...
    des.utilAi, des.utilSp, desWqAi, desWqSp, desRespMin, ...
    des.avgQs, des.maxQs, ...
    'VariableNames', {'arrivalsPerDay', 'lambdaPerSec', 'aiMeanS', 'bwMeanS', ...
    'specMeanS', 'utilAiAn', 'utilSpAn', 'WqAiAn', 'WqSpAn', 'respMinAn', ...
    'utilAiDes', 'utilSpDes', 'WqAiDes', 'WqSpDes', 'respMinDes', ...
    'avgSpQDes', 'maxSpQDes'});
report.text = char(join(lines, newline));
end

% --- formatting helpers ---------------------------------------------------------
function s = fmtPct(x)
if isinf(x), s = '>99%'; else, s = sprintf('%.1f%%', 100 * x); end
end
function s = fmtSec(x)
if isinf(x), s = 'inf'; else, s = sprintf('%.1f', x); end
end
function s = fmtMin(x)
if isinf(x), s = 'inf'; else, s = sprintf('%.2f', x / 60); end
end
function s = fmtVal(x)
if isnan(x), s = 'n/a'; else, s = sprintf('%.2f', x); end
end
function Lq = mm1LambdaWq(Wq, lambda)
Lq = Wq * lambda;
end

% --- CSV output ------------------------------------------------------------------
function writeReport(csvFile, report)
try
    writetable(report.csvRows, csvFile);
    fprintf('[telemedicine_sim] Report written to %s\n', csvFile);
catch err %#ok<NASGU>
    fprintf('[telemedicine_sim] Could not write %s\n', csvFile);
end
end