/* =====================================================================
   Churn Prediction API — script.js
   ---------------------------------------------------------------------
   1. Motor do modelo (Regressão Logística exportada do scikit-learn)
   2. "FastAPI no navegador": mesmas rotas, validação Pydantic (422),
      regras de negócio (400) e respostas JSON da API Python
   3. Cliente para a API FastAPI real (fetch)
   4. Gráficos SVG (velocímetro, linha, barras, histograma, rosca)
   5. Interface: predição, what-if, lote, laboratório da API
   6. Página: tema, galeria, lightbox, copiar, animações
   ===================================================================== */
(function (global) {
  "use strict";

  /* ===================================================================
     1. MOTOR DO MODELO
     Reproduz o pipeline: imputação -> StandardScaler -> OneHotEncoder
     -> LogisticRegression.predict_proba
     =================================================================== */
  var ChurnEngine = (function () {
    function model() { return global.CHURN_MODEL; }

    function isMissing(v) {
      return v === null || v === undefined || v === "" || (typeof v === "number" && isNaN(v));
    }

    /** Termos do log-odds: [{feature, value, term}] */
    function terms(rec) {
      var m = model(), out = [];
      m.numeric.forEach(function (f) {
        var v = isMissing(rec[f.name]) ? f.median : Number(rec[f.name]);
        var z = (v - f.mean) / f.scale;
        out.push({ feature: f.name, value: v, term: f.coef * z, kind: "num" });
      });
      m.categorical.forEach(function (f) {
        var v = isMissing(rec[f.name]) ? f.mode : String(rec[f.name]);
        var c = Object.prototype.hasOwnProperty.call(f.coefs, v) ? f.coefs[v] : 0; // handle_unknown="ignore"
        out.push({ feature: f.name, value: v, term: c, kind: "cat" });
      });
      return out;
    }

    function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

    function predictProba(rec) {
      var logit = model().intercept;
      terms(rec).forEach(function (t) { logit += t.term; });
      return sigmoid(logit);
    }

    /** Contribuição de cada variável em relação a um cliente "médio".
        Numéricas: coef * z (z=0 é a média). Categóricas: coef da categoria
        menos a média dos coeficientes da variável. */
    function explain(rec) {
      var m = model(), byName = {};
      m.categorical.forEach(function (f) {
        var vals = Object.keys(f.coefs).map(function (k) { return f.coefs[k]; });
        byName[f.name] = vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
      });
      return terms(rec).map(function (t) {
        var contrib = t.kind === "num" ? t.term : t.term - byName[t.feature];
        return { feature: t.feature, value: t.value, contribution: contrib };
      }).sort(function (a, b) { return Math.abs(b.contribution) - Math.abs(a.contribution); });
    }

    function riskLevel(p) { return p < 0.3 ? "baixo" : p < 0.6 ? "medio" : "alto"; }

    return { predictProba: predictProba, explain: explain, riskLevel: riskLevel };
  })();

  /* ===================================================================
     2. "FASTAPI NO NAVEGADOR"
     Espelha app/schemas/churn.py, app/services/model_service.py e as rotas.
     =================================================================== */
  var YES_NO = ["Yes", "No"];
  var YES_NO_NET = ["Yes", "No", "No internet service"];
  var SCHEMA = {
    gender: { lit: ["Male", "Female"] },
    SeniorCitizen: { lit: YES_NO },
    Partner: { lit: YES_NO },
    Dependents: { lit: YES_NO },
    tenure: { type: "int", ge: 0, le: 120 },
    PhoneService: { lit: YES_NO },
    MultipleLines: { lit: ["Yes", "No", "No phone service"] },
    InternetService: { lit: ["DSL", "Fiber optic", "No"] },
    OnlineSecurity: { lit: YES_NO_NET },
    OnlineBackup: { lit: YES_NO_NET },
    DeviceProtection: { lit: YES_NO_NET },
    TechSupport: { lit: YES_NO_NET },
    StreamingTV: { lit: YES_NO_NET },
    StreamingMovies: { lit: YES_NO_NET },
    Contract: { lit: ["Month-to-month", "One year", "Two year"] },
    PaperlessBilling: { lit: YES_NO },
    PaymentMethod: { lit: ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] },
    MonthlyCharges: { type: "float", gt: 0, le: 1000 },
    TotalCharges: { type: "float", ge: 0, le: 100000 }
  };
  var FEATURES = Object.keys(SCHEMA);
  var ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"];
  var MAX_BATCH = 100;
  var API_VERSION = "1.0.0";

  function litMsg(opts) {
    var q = opts.map(function (o) { return "'" + o + "'"; });
    return "Input should be " + (q.length === 2 ? q.join(" or ") : q.slice(0, -1).join(", ") + " or " + q[q.length - 1]);
  }

  function validateField(name, raw, loc, errors) {
    var s = SCHEMA[name], v = raw;
    function err(type, msg, ctx) {
      var e = { type: type, loc: loc.concat([name]), msg: msg, input: raw };
      if (ctx) e.ctx = ctx;
      errors.push(e);
    }
    if (s.lit) {
      if (s.lit.indexOf(raw) === -1) { err("literal_error", litMsg(s.lit), { expected: s.lit.map(function (o) { return "'" + o + "'"; }).join(", ") }); return undefined; }
      return raw;
    }
    if (s.type === "int") {
      if (typeof raw === "number") {
        if (!isFinite(raw)) { err("finite_number", "Input should be a finite number"); return undefined; }
        if (Math.floor(raw) !== raw) { err("int_from_float", "Input should be a valid integer, got a number with a fractional part"); return undefined; }
      } else if (typeof raw === "string") {
        if (!/^\s*[-+]?\d+\s*$/.test(raw)) { err("int_parsing", "Input should be a valid integer, unable to parse string as an integer"); return undefined; }
        v = parseInt(raw, 10);
      } else { err("int_type", "Input should be a valid integer"); return undefined; }
    } else {
      if (typeof raw === "string") {
        if (raw.trim() === "" || isNaN(Number(raw))) { err("float_parsing", "Input should be a valid number, unable to parse string as a number"); return undefined; }
        v = Number(raw);
      } else if (typeof raw !== "number") { err("float_type", "Input should be a valid number"); return undefined; }
    }
    if (s.gt !== undefined && !(v > s.gt)) { err("greater_than", "Input should be greater than " + s.gt, { gt: s.gt }); return undefined; }
    if (s.ge !== undefined && !(v >= s.ge)) { err("greater_than_equal", "Input should be greater than or equal to " + s.ge, { ge: s.ge }); return undefined; }
    if (s.le !== undefined && !(v <= s.le)) { err("less_than_equal", "Input should be less than or equal to " + s.le, { le: s.le }); return undefined; }
    return v;
  }

  /** Valida um CustomerFeatures (extra="forbid"). */
  function validateCustomer(obj, loc, errors) {
    if (obj === null || typeof obj !== "object" || Array.isArray(obj)) {
      errors.push({ type: "model_attributes_type", loc: loc, msg: "Input should be a valid dictionary or object to extract fields from", input: obj });
      return null;
    }
    var clean = {};
    FEATURES.forEach(function (f) {
      if (!Object.prototype.hasOwnProperty.call(obj, f)) {
        errors.push({ type: "missing", loc: loc.concat([f]), msg: "Field required", input: obj });
      } else {
        clean[f] = validateField(f, obj[f], loc, errors);
      }
    });
    Object.keys(obj).forEach(function (k) {
      if (!SCHEMA[k]) errors.push({ type: "extra_forbidden", loc: loc.concat([k]), msg: "Extra inputs are not permitted", input: obj[k] });
    });
    return clean;
  }

  /** Regras de negócio (HTTP 400) — mesmas de model_service.py */
  function businessRule(c) {
    var addons = ADDONS.map(function (a) { return c[a]; });
    if (c.InternetService === "No" && addons.some(function (v) { return v !== "No internet service"; }))
      return "Cliente sem internet deve ter os serviços online como 'No internet service'.";
    if (c.InternetService !== "No" && addons.some(function (v) { return v === "No internet service"; }))
      return "Cliente com internet não pode ter serviço marcado como 'No internet service'.";
    if (c.PhoneService === "No" && c.MultipleLines !== "No phone service")
      return "Cliente sem telefonia deve ter MultipleLines = 'No phone service'.";
    if (c.TotalCharges < c.MonthlyCharges * c.tenure * 0.5)
      return "TotalCharges incoerente: muito abaixo de MonthlyCharges × tenure.";
    return null;
  }

  function predictionResponse(c) {
    var p = ChurnEngine.predictProba(c), m = global.CHURN_MODEL;
    return {
      churn_prediction: p >= m.threshold,
      churn_probability: Math.round(p * 10000) / 10000,
      risk_level: ChurnEngine.riskLevel(p),
      model_version: m.model_version
    };
  }

  var LocalApi = {
    label: "navegador",
    /** request(method, path, body) -> Promise<{ok,status,data,ms,error}> */
    request: function (method, path, body) {
      var t0 = performance.now();
      var res = LocalApi._handle(String(method).toUpperCase(), path, body);
      res.ms = Math.max(0.01, performance.now() - t0);
      res.ok = res.status >= 200 && res.status < 300;
      res.error = res.ok ? null : formatDetail(res.data && res.data.detail);
      return Promise.resolve(res);
    },
    _handle: function (method, path, body) {
      var m = global.CHURN_MODEL;
      var routes = { "/": "GET", "/health": "GET", "/v1/model/info": "GET", "/v1/predict": "POST", "/v1/predict/batch": "POST" };
      path = (path || "/").split("?")[0];
      if (!routes[path]) return { status: 404, data: { detail: "Not Found" } };
      if (routes[path] !== method) return { status: 405, data: { detail: "Method Not Allowed" } };

      if (path === "/") return { status: 200, data: { api: "Churn Prediction API", version: API_VERSION, docs: "/docs", health: "/health" } };
      if (path === "/health") return { status: 200, data: { status: "ok", model_loaded: true, model_name: m.model_name, model_version: m.model_version, api_version: API_VERSION } };
      if (path === "/v1/model/info") return { status: 200, data: global.CHURN_METRICS };

      // corpo JSON: aceita objeto ou texto
      if (typeof body === "string") {
        try { body = JSON.parse(body); }
        catch (e) { return { status: 422, data: { detail: [{ type: "json_invalid", loc: ["body", 0], msg: "JSON decode error", input: {}, ctx: { error: e.message } }] } }; }
      }
      var errors = [];
      if (path === "/v1/predict") {
        var c = validateCustomer(body, ["body"], errors);
        if (errors.length) return { status: 422, data: { detail: errors } };
        var rule = businessRule(c);
        if (rule) return { status: 400, data: { detail: rule } };
        return { status: 200, data: predictionResponse(c) };
      }
      // /v1/predict/batch
      if (body === null || typeof body !== "object" || Array.isArray(body))
        return { status: 422, data: { detail: [{ type: "model_attributes_type", loc: ["body"], msg: "Input should be a valid dictionary or object to extract fields from", input: body }] } };
      Object.keys(body).forEach(function (k) {
        if (k !== "customers") errors.push({ type: "extra_forbidden", loc: ["body", k], msg: "Extra inputs are not permitted", input: body[k] });
      });
      var list = body.customers;
      if (list === undefined) errors.push({ type: "missing", loc: ["body", "customers"], msg: "Field required", input: body });
      else if (!Array.isArray(list)) errors.push({ type: "list_type", loc: ["body", "customers"], msg: "Input should be a valid list", input: list });
      else if (list.length < 1) errors.push({ type: "too_short", loc: ["body", "customers"], msg: "List should have at least 1 item after validation, not 0", input: list, ctx: { field_type: "List", min_length: 1, actual_length: 0 } });
      else if (list.length > MAX_BATCH) errors.push({ type: "too_long", loc: ["body", "customers"], msg: "List should have at most " + MAX_BATCH + " items after validation, not " + list.length, input: "[…]", ctx: { field_type: "List", max_length: MAX_BATCH, actual_length: list.length } });
      var cleaned = [];
      if (!errors.length) list.forEach(function (item, i) { cleaned.push(validateCustomer(item, ["body", "customers", i], errors)); });
      if (errors.length) return { status: 422, data: { detail: errors } };
      for (var i = 0; i < cleaned.length; i++) {
        var r = businessRule(cleaned[i]);
        if (r) return { status: 400, data: { detail: r } };
      }
      var preds = cleaned.map(predictionResponse);
      return { status: 200, data: { total: preds.length, predictions: preds } };
    }
  };

  /** Erro 422 do Pydantic (lista) -> texto legível, como no api_client.py */
  function formatDetail(detail) {
    if (Array.isArray(detail)) {
      return detail.map(function (e) {
        var loc = (e.loc || []).filter(function (p) { return p !== "body"; }).join(".");
        return (loc ? loc + ": " : "") + e.msg;
      }).join(" | ");
    }
    return detail === undefined ? "Erro desconhecido" : String(detail);
  }

  /* ===================================================================
     3. CLIENTE DA API FASTAPI REAL
     =================================================================== */
  function RemoteApi(baseUrl) {
    this.base = String(baseUrl || "").replace(/\/+$/, "");
    this.label = "API real";
  }
  RemoteApi.prototype.request = function (method, path, body) {
    var t0 = performance.now(), opts = { method: method, headers: { Accept: "application/json" } };
    if (body !== undefined && body !== null && method !== "GET") {
      opts.headers["Content-Type"] = "application/json";
      opts.body = typeof body === "string" ? body : JSON.stringify(body);
    }
    var ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    if (ctrl) { opts.signal = ctrl.signal; setTimeout(function () { ctrl.abort(); }, 15000); }
    return fetch(this.base + path, opts).then(function (resp) {
      return resp.text().then(function (txt) {
        var data; try { data = JSON.parse(txt); } catch (e) { data = { detail: txt }; }
        var ok = resp.ok;
        return { ok: ok, status: resp.status, data: data, ms: performance.now() - t0, error: ok ? null : formatDetail(data && data.detail) };
      });
    }).catch(function (e) {
      return { ok: false, status: null, data: null, ms: performance.now() - t0,
        error: "Não foi possível conectar em " + this.base + ". Verifique se a API está rodando e com CORS liberado." };
    }.bind(this));
  };

  /** Divide em lotes de 100 (limite da API), como predict_many do api_client.py */
  function predictMany(api, customers, onProgress) {
    var preds = [], totalMs = 0, chunks = [];
    for (var i = 0; i < customers.length; i += MAX_BATCH) chunks.push(customers.slice(i, i + MAX_BATCH));
    var idx = 0;
    function next() {
      if (idx >= chunks.length) return Promise.resolve({ ok: true, status: 200, data: { total: preds.length, predictions: preds }, ms: totalMs, requests: chunks.length });
      var start = idx * MAX_BATCH;
      return api.request("POST", "/v1/predict/batch", { customers: chunks[idx] }).then(function (r) {
        if (!r.ok) { r.error = "Lote " + (idx + 1) + " (linhas " + (start + 1) + "–" + (start + chunks[idx].length) + "): " + r.error; return r; }
        preds = preds.concat(r.data.predictions); totalMs += r.ms; idx++;
        if (onProgress) onProgress(idx / chunks.length);
        return next();
      });
    }
    return next();
  }

  // Exporta para testes em Node
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { ChurnEngine: ChurnEngine, LocalApi: LocalApi, formatDetail: formatDetail };
  }
  if (typeof document === "undefined") return;

  /* ===================================================================
     4. GRÁFICOS SVG (sem bibliotecas externas)
     =================================================================== */
  var NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(n);
    return n;
  }
  function svgRoot(w, h, label) {
    var s = el("svg", { viewBox: "0 0 " + w + " " + h, role: "img", "aria-label": label || "", "class": "chart" });
    return s;
  }
  function txt(parent, x, y, s, cls, anchor) {
    var t = el("text", { x: x, y: y, "class": cls || "ch-label", "text-anchor": anchor || "middle" }, parent);
    t.textContent = s; return t;
  }
  function tip(node, s) { var t = el("title", {}, node); t.textContent = s; }
  function riskClass(p) { return "r-" + ChurnEngine.riskLevel(p); }
  function pct(p, d) { return (p * 100).toLocaleString("pt-BR", { minimumFractionDigits: d === undefined ? 1 : d, maximumFractionDigits: d === undefined ? 1 : d }) + "%"; }
  function num(v, d) { return Number(v).toLocaleString("pt-BR", { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 }); }

  function arcPath(cx, cy, r, a0, a1) {
    var x0 = cx + r * Math.cos(a0), y0 = cy - r * Math.sin(a0), x1 = cx + r * Math.cos(a1), y1 = cy - r * Math.sin(a1);
    return "M" + x0 + " " + y0 + " A" + r + " " + r + " 0 0 1 " + x1 + " " + y1;
  }
  function Gauge(p) {
    var s = svgRoot(300, 180, "Probabilidade de churn: " + pct(p)), cx = 150, cy = 150, r = 112;
    var bands = [[0, 0.3, "r-baixo"], [0.3, 0.6, "r-medio"], [0.6, 1, "r-alto"]];
    bands.forEach(function (b) {
      el("path", { d: arcPath(cx, cy, r, Math.PI * (1 - b[0]), Math.PI * (1 - b[1])), "class": "g-band " + b[2] }, s);
    });
    el("path", { d: arcPath(cx, cy, r, Math.PI, Math.PI * (1 - Math.max(0.002, p))), "class": "g-value " + riskClass(p) }, s);
    var a = Math.PI * 0.5; // marcador do limiar 50%
    el("line", { x1: cx + (r - 22) * Math.cos(a), y1: cy - (r - 22) * Math.sin(a), x2: cx + (r + 18) * Math.cos(a), y2: cy - (r + 18) * Math.sin(a), "class": "g-tick" }, s);
    txt(s, cx, cy - (r + 24), "limiar 50%", "ch-label sm");
    txt(s, cx, cy - 16, pct(p), "g-num");
    txt(s, cx - r, cy + 22, "0%", "ch-label sm"); txt(s, cx + r, cy + 22, "100%", "ch-label sm");
    return s;
  }

  function niceCeil(v) {
    if (v <= 0) return 1;
    var step = Math.pow(10, Math.floor(Math.log10(v))), m = v / step;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * step;
  }
  /** Largura do viewBox de acordo com o espaço real (textos legíveis no celular). */
  function fitW(box, max) { var w = box && box.clientWidth ? box.clientWidth - 8 : max; return Math.max(320, Math.min(max, w)); }
  function frame(w, h, m) { return { w: w, h: h, m: m, iw: w - m.l - m.r, ih: h - m.t - m.b }; }
  function yAxis(s, f, max, ticks, fmt) {
    for (var i = 0; i <= ticks; i++) {
      var v = max * i / ticks, y = f.m.t + f.ih - f.ih * i / ticks;
      el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y, y2: y, "class": "ch-grid" }, s);
      txt(s, f.m.l - 8, y + 4, fmt(v), "ch-label", "end");
    }
  }
  function riskBands(s, f) {
    [[0, 30, "r-baixo"], [30, 60, "r-medio"], [60, 100, "r-alto"]].forEach(function (b) {
      var y0 = f.m.t + f.ih - f.ih * b[1] / 100, y1 = f.m.t + f.ih - f.ih * b[0] / 100;
      el("rect", { x: f.m.l, y: y0, width: f.iw, height: y1 - y0, "class": "ch-band " + b[2] }, s);
    });
  }

  function LineChart(xs, probs, xTitle, current, fmtX, w) {
    var f = frame(w || 720, 320, { t: 20, r: 20, b: 52, l: 52 }), s = svgRoot(f.w, f.h, "Risco por " + xTitle);
    var x0 = xs[0], x1 = xs[xs.length - 1];
    function X(v) { return f.m.l + (v - x0) / (x1 - x0) * f.iw; }
    function Y(p) { return f.m.t + f.ih - p * f.ih; }
    riskBands(s, f); yAxis(s, f, 100, 5, function (v) { return v + "%"; });
    var nt = f.w < 500 ? 4 : 6; for (var i = 0; i <= nt; i++) { var v = x0 + (x1 - x0) * i / nt; txt(s, X(v), f.h - 30, fmtX(v), "ch-label"); }
    txt(s, f.m.l + f.iw / 2, f.h - 6, xTitle, "ch-title");
    if (current >= x0 && current <= x1) {
      el("line", { x1: X(current), x2: X(current), y1: f.m.t, y2: f.m.t + f.ih, "class": "ch-current" }, s);
      txt(s, X(current) + 6, f.m.t + 12, "cliente atual", "ch-label sm", "start");
    }
    el("path", { d: xs.map(function (x, i) { return (i ? "L" : "M") + X(x) + " " + Y(probs[i]); }).join(" "), "class": "ch-line" }, s);
    xs.forEach(function (x, i) {
      var c = el("circle", { cx: X(x), cy: Y(probs[i]), r: 4, "class": "ch-dot" }, s);
      tip(c, xTitle + ": " + fmtX(x) + " → churn " + pct(probs[i]));
    });
    return s;
  }

  function BarChart(labels, values, opts) {
    opts = opts || {};
    var f = frame(opts.w || 720, opts.h || 300, { t: 28, r: 12, b: opts.b || 56, l: 46 }), s = svgRoot(f.w, f.h, opts.label || "");
    var max = opts.max || niceCeil(Math.max.apply(null, values) * 1.15);
    if (opts.bands) riskBands(s, f);
    yAxis(s, f, max, 4, opts.fmtY || function (v) { return Math.round(v); });
    var bw = f.iw / labels.length;
    labels.forEach(function (lab, i) {
      var v = values[i], h = f.ih * v / max, x = f.m.l + i * bw + bw * 0.18, w = bw * 0.64, y = f.m.t + f.ih - h;
      var r = el("rect", { x: x, y: y, width: w, height: Math.max(0, h), rx: 4, "class": "ch-bar " + (opts.cls ? opts.cls(i, v) : "") }, s);
      tip(r, lab + ": " + (opts.fmtV ? opts.fmtV(v) : v));
      txt(s, x + w / 2, y - 7, opts.fmtV ? opts.fmtV(v) : v, "ch-value");
      wrapLabel(s, x + w / 2, f.m.t + f.ih + 18, lab, Math.max(8, Math.floor(bw / 7.5)));
    });
    return s;
  }
  function wrapLabel(s, x, y, label, maxChars) {
    var words = String(label).split(" "), lines = [""];
    words.forEach(function (w) {
      var cur = lines[lines.length - 1];
      if ((cur + " " + w).trim().length > maxChars && cur) lines.push(w); else lines[lines.length - 1] = (cur + " " + w).trim();
    });
    lines.slice(0, 2).forEach(function (l, i) { txt(s, x, y + i * 15, l, "ch-label"); });
  }

  function Histogram(probs, w) {
    var bins = new Array(20).fill(0);
    probs.forEach(function (p) { bins[Math.min(19, Math.floor(p * 20))]++; });
    var f = frame(w || 720, 300, { t: 28, r: 16, b: 52, l: 46 }), s = svgRoot(f.w, f.h, "Distribuição das probabilidades");
    var max = niceCeil(Math.max.apply(null, bins) * 1.1);
    yAxis(s, f, max, 4, function (v) { return Math.round(v); });
    var bw = f.iw / 20;
    bins.forEach(function (n, i) {
      var h = f.ih * n / max, x = f.m.l + i * bw + 1, y = f.m.t + f.ih - h;
      var r = el("rect", { x: x, y: y, width: bw - 2, height: h, rx: 2, "class": "ch-bar " + riskClass((i + 0.5) / 20) }, s);
      tip(r, (i * 5) + "–" + (i * 5 + 5) + "%: " + n + " clientes");
    });
    for (var t = 0; t <= 100; t += 20) txt(s, f.m.l + f.iw * t / 100, f.h - 30, t + "%", "ch-label");
    var xt = f.m.l + f.iw / 2;
    el("line", { x1: xt, x2: xt, y1: f.m.t - 8, y2: f.m.t + f.ih, "class": "ch-current" }, s);
    txt(s, xt + 6, f.m.t - 2, "limiar 50%", "ch-label sm", "start");
    txt(s, f.m.l + f.iw / 2, f.h - 6, "Probabilidade de churn", "ch-title");
    return s;
  }

  function Donut(counts) {
    var order = ["baixo", "medio", "alto"], total = order.reduce(function (a, k) { return a + (counts[k] || 0); }, 0);
    var s = svgRoot(260, 260, "Clientes por faixa de risco"), cx = 130, cy = 130, r = 96, a = Math.PI / 2;
    order.forEach(function (k) {
      var n = counts[k] || 0; if (!n) return;
      var frac = n / total, a1 = a - frac * 2 * Math.PI;
      var large = frac > 0.5 ? 1 : 0;
      var x0 = cx + r * Math.cos(a), y0 = cy - r * Math.sin(a), x1 = cx + r * Math.cos(a1), y1 = cy - r * Math.sin(a1);
      var d = frac >= 0.9999 ? "M" + cx + " " + (cy - r) + " A" + r + " " + r + " 0 1 1 " + (cx - 0.01) + " " + (cy - r)
        : "M" + x0 + " " + y0 + " A" + r + " " + r + " 0 " + large + " 1 " + x1 + " " + y1;
      var p = el("path", { d: d, "class": "dn-seg r-" + k }, s);
      tip(p, RISK_LABEL[k] + ": " + n + " clientes (" + pct(frac) + ")");
      a = a1;
    });
    txt(s, cx, cy - 2, num(total), "dn-num"); txt(s, cx, cy + 20, "clientes", "ch-label");
    return s;
  }

  /* ===================================================================
     5. INTERFACE DA DEMO
     =================================================================== */
  var LABELS = {
    Yes: "Sim", No: "Não", Male: "Masculino", Female: "Feminino",
    "No internet service": "Sem internet", "No phone service": "Sem telefonia",
    DSL: "DSL", "Fiber optic": "Fibra óptica",
    "Month-to-month": "Mensal", "One year": "Anual", "Two year": "Bienal",
    "Electronic check": "Boleto eletrônico", "Mailed check": "Cheque pelo correio",
    "Bank transfer (automatic)": "Débito automático", "Credit card (automatic)": "Cartão (automático)"
  };
  var FEATURE_LABEL = {
    gender: "Gênero", SeniorCitizen: "Idoso (65+)", Partner: "Parceiro(a)", Dependents: "Dependentes",
    tenure: "Meses como cliente", PhoneService: "Telefonia", MultipleLines: "Múltiplas linhas",
    InternetService: "Internet", OnlineSecurity: "Segurança online", OnlineBackup: "Backup online",
    DeviceProtection: "Proteção de dispositivo", TechSupport: "Suporte técnico", StreamingTV: "Streaming de TV",
    StreamingMovies: "Streaming de filmes", Contract: "Contrato", PaperlessBilling: "Fatura digital",
    PaymentMethod: "Pagamento", MonthlyCharges: "Mensalidade", TotalCharges: "Total faturado"
  };
  var RISK_LABEL = { baixo: "Baixo", medio: "Médio", alto: "Alto" };
  function L(v) { return LABELS[v] || v; }
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function h(tag, attrs, html) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { if (k === "class") n.className = attrs[k]; else n.setAttribute(k, attrs[k]); });
    if (html !== undefined) n.innerHTML = html;
    return n;
  }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function fmtMs(ms) { return ms < 1 ? ms.toLocaleString("pt-BR", { maximumFractionDigits: 2 }) + " ms" : num(ms, ms < 100 ? 1 : 0) + " ms"; }
  function jsonHtml(obj) {
    return esc(JSON.stringify(obj, null, 2))
      .replace(/(&quot;[^&]*?&quot;)(\s*:)/g, '<span class="k">$1</span>$2')
      .replace(/:\s(&quot;.*?&quot;)/g, ': <span class="s">$1</span>');
  }
  function statusClass(st) { return !st ? "st-err" : st < 300 ? "st-ok" : st < 500 ? "st-warn" : "st-err"; }

  var DEFAULT_PROFILE = {
    gender: "Female", SeniorCitizen: "No", Partner: "Yes", Dependents: "No", tenure: 2,
    PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic",
    OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No",
    StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check", MonthlyCharges: 95.7, TotalCharges: 191.4
  };
  var LOYAL_PROFILE = Object.assign({}, DEFAULT_PROFILE, {
    tenure: 60, Contract: "Two year", PaymentMethod: "Credit card (automatic)", InternetService: "DSL",
    OnlineSecurity: "Yes", TechSupport: "Yes", MonthlyCharges: 60.0, TotalCharges: 3600.0
  });

  var state = { api: LocalApi, profile: Object.assign({}, DEFAULT_PROFILE), autoTotal: true, sample: null, batch: null, whatif: "tenure" };

  /* ---------- CSV ---------- */
  function parseCSV(text) {
    var rows = [], row = [], field = "", q = false;
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      if (q) {
        if (ch === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; }
        else field += ch;
      } else if (ch === '"') q = true;
      else if (ch === ",") { row.push(field); field = ""; }
      else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && text[i + 1] === "\n") i++;
        row.push(field); field = ""; if (row.length > 1 || row[0] !== "") rows.push(row); row = [];
      } else field += ch;
    }
    if (field !== "" || row.length) { row.push(field); rows.push(row); }
    var head = rows.shift().map(function (x) { return x.trim(); });
    return { columns: head, rows: rows.map(function (r) { var o = {}; head.forEach(function (c, j) { o[c] = r[j] === undefined ? "" : r[j].trim(); }); return o; }) };
  }
  function toPayload(row) {
    var o = {};
    FEATURES.forEach(function (f) {
      var v = row[f];
      if (f === "SeniorCitizen" && (v === "0" || v === "1")) v = v === "1" ? "Yes" : "No";
      if (f === "tenure") v = parseInt(v, 10);
      if (f === "MonthlyCharges") v = parseFloat(v);
      if (f === "TotalCharges") { v = parseFloat(v); if (isNaN(v)) v = 0; }
      o[f] = v;
    });
    return o;
  }

  /* ---------- Conexão (motor) ---------- */
  function setStatus(kind, text) {
    var s = $("#apiStatus"); s.className = "api-status " + kind; s.querySelector("span").textContent = text;
  }
  function checkHealth() {
    setStatus("pending", "Verificando…");
    return state.api.request("GET", "/health").then(function (r) {
      if (r.ok && r.data.model_loaded) setStatus("ok", (state.api === LocalApi ? "Modelo no navegador" : "API online") + " · " + fmtMs(r.ms));
      else if (r.ok) setStatus("warn", "API online · modelo indisponível");
      else setStatus("err", r.error || "Offline");
      return r;
    });
  }
  function initEngine() {
    var radios = $all('input[name="engine"]'), urlBox = $("#apiUrlBox"), url = $("#apiUrl");
    try { var saved = localStorage.getItem("apiUrl"); if (saved) url.value = saved; } catch (e) {}
    function apply() {
      var remote = $('input[name="engine"]:checked').value === "remote";
      urlBox.hidden = !remote;
      state.api = remote ? new RemoteApi(url.value) : LocalApi;
      $all(".engine-name").forEach(function (n) { n.textContent = remote ? "API FastAPI (" + url.value + ")" : "FastAPI simulada no navegador"; });
      checkHealth().then(refreshAll);
    }
    radios.forEach(function (r) { r.addEventListener("change", apply); });
    $("#apiConnect").addEventListener("click", function () {
      try { localStorage.setItem("apiUrl", url.value); } catch (e) {}
      apply();
    });
    apply();
  }

  /* ---------- Tabs ---------- */
  function initTabs() {
    $all(".demo-tab").forEach(function (t) {
      t.addEventListener("click", function () {
        $all(".demo-tab").forEach(function (o) { o.setAttribute("aria-selected", o === t ? "true" : "false"); });
        $all(".demo-panel").forEach(function (p) { p.hidden = p.id !== t.getAttribute("aria-controls"); });
        if (t.dataset.panel === "whatif") renderWhatIf();
        if (t.dataset.panel === "batch" && !state.batch) $("#batchEmpty").hidden = false;
      });
    });
  }

  /* ---------- Formulário (reativo, como o Streamlit) ---------- */
  var SELECTS = {
    gender: ["Female", "Male"], SeniorCitizen: ["No", "Yes"], Partner: ["Yes", "No"], Dependents: ["No", "Yes"],
    PhoneService: ["Yes", "No"], InternetService: ["Fiber optic", "DSL", "No"],
    Contract: ["Month-to-month", "One year", "Two year"],
    PaymentMethod: ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    PaperlessBilling: ["Yes", "No"]
  };
  function fillSelect(sel, opts, value, locked) {
    sel.innerHTML = "";
    opts.forEach(function (o) { var op = h("option", { value: o }, esc(L(o))); if (o === value) op.selected = true; sel.appendChild(op); });
    sel.disabled = !!locked;
  }
  function syncForm() {
    var p = state.profile, f = $("#custForm");
    Object.keys(SELECTS).forEach(function (k) { fillSelect(f.elements[k], SELECTS[k], p[k]); });
    if (p.PhoneService === "No") { p.MultipleLines = "No phone service"; fillSelect(f.elements.MultipleLines, ["No phone service"], p.MultipleLines, true); }
    else { if (p.MultipleLines === "No phone service") p.MultipleLines = "No"; fillSelect(f.elements.MultipleLines, ["No", "Yes"], p.MultipleLines); }
    ADDONS.forEach(function (a) {
      if (p.InternetService === "No") { p[a] = "No internet service"; fillSelect(f.elements[a], ["No internet service"], p[a], true); }
      else { if (p[a] === "No internet service") p[a] = "No"; fillSelect(f.elements[a], ["No", "Yes"], p[a]); }
    });
    f.elements.tenure.value = p.tenure; $("#tenureOut").textContent = p.tenure + (p.tenure === 1 ? " mês" : " meses");
    f.elements.MonthlyCharges.value = p.MonthlyCharges;
    f.elements.autoTotal.checked = state.autoTotal;
    f.elements.TotalCharges.disabled = state.autoTotal;
    if (state.autoTotal) p.TotalCharges = Math.round(p.MonthlyCharges * p.tenure * 100) / 100;
    f.elements.TotalCharges.value = p.TotalCharges;
  }
  function readForm() {
    var f = $("#custForm"), p = state.profile;
    FEATURES.forEach(function (k) {
      var e = f.elements[k]; if (!e) return;
      if (k === "tenure") p[k] = parseInt(e.value, 10) || 0;
      else if (k === "MonthlyCharges" || k === "TotalCharges") { var v = parseFloat(e.value); p[k] = isNaN(v) ? 0 : v; }
      else p[k] = e.value;
    });
    state.autoTotal = f.elements.autoTotal.checked;
  }
  var predictTimer = null;
  function initForm() {
    var f = $("#custForm");
    f.addEventListener("input", function (ev) {
      readForm(); syncForm();
      clearTimeout(predictTimer); predictTimer = setTimeout(runPredict, ev.target.type === "range" ? 60 : 120);
    });
    f.addEventListener("submit", function (ev) { ev.preventDefault(); runPredict(); });
    $all("[data-preset]").forEach(function (b) {
      b.addEventListener("click", function () {
        var k = b.dataset.preset;
        if (k === "risk") state.profile = Object.assign({}, DEFAULT_PROFILE);
        else if (k === "loyal") state.profile = Object.assign({}, LOYAL_PROFILE);
        else {
          var rows = getSample().rows, r = rows[Math.floor(Math.random() * rows.length)];
          state.profile = toPayload(r);
          state.autoTotal = false;
        }
        if (k !== "random") state.autoTotal = true;
        syncForm(); runPredict();
      });
    });
    syncForm();
    // barra fixa com o risco atual no celular, enquanto o formulário está na tela
    if ("IntersectionObserver" in window) {
      var mini = $("#predMini"), seeForm = false, seeRes = false;
      var upd = function () { mini.classList.toggle("show", seeForm && !seeRes && !$("#panel-predict").hidden); };
      new IntersectionObserver(function (es) { seeForm = es[0].isIntersecting; upd(); }, { rootMargin: "0px 0px -40% 0px" }).observe(f);
      new IntersectionObserver(function (es) { seeRes = es[0].isIntersecting; upd(); }, { threshold: 0.15 }).observe($("#predResult"));
    }
  }

  function retentionActions(c) {
    var a = [];
    if (c.Contract === "Month-to-month") a.push("Oferecer migração para plano <b>anual</b> com desconto.");
    if (c.PaymentMethod === "Electronic check") a.push("Incentivar <b>pagamento automático</b> (débito ou cartão).");
    if (c.InternetService !== "No" && c.TechSupport === "No") a.push("Incluir <b>suporte técnico</b> gratuito por 3 meses.");
    if (c.InternetService !== "No" && c.OnlineSecurity === "No") a.push("Oferecer <b>segurança online</b> como bônus de fidelidade.");
    if (c.tenure < 12) a.push("Acionar programa de <b>onboarding</b> para clientes novos.");
    if (c.MonthlyCharges > 80) a.push("Revisar o plano: mensalidade <b>acima da média</b> da base.");
    return a.length ? a : ["Perfil estável: manter relacionamento e pesquisa de satisfação."];
  }

  function curlFor(method, path, body) {
    var base = state.api === LocalApi ? "http://127.0.0.1:8000" : state.api.base;
    var c = "curl -X " + method + " " + base + path;
    if (body !== undefined && method !== "GET") c += " \\\n  -H 'Content-Type: application/json' \\\n  -d '" + (typeof body === "string" ? body : JSON.stringify(body)) + "'";
    return c;
  }

  function runPredict() {
    var payload = Object.assign({}, state.profile), out = $("#predResult");
    out.classList.add("loading");
    state.api.request("POST", "/v1/predict", payload).then(function (r) {
      out.classList.remove("loading");
      if (!r.ok) {
        out.innerHTML = '<div class="alert err"><b>HTTP ' + (r.status || "—") + "</b> · " + esc(r.error) + "</div>";
        return;
      }
      var d = r.data, p = d.churn_probability;
      out.innerHTML = "";
      $("#predMiniP").textContent = pct(p);
      $("#predMiniR").innerHTML = '<span class="risk-badge ' + riskClass(p) + '" style="font-size:12px;padding:3px 10px">Risco ' + RISK_LABEL[d.risk_level] + "</span>";
      var card = h("div", { class: "res-card" });
      card.appendChild(Gauge(p));
      card.appendChild(h("div", { class: "res-badge-wrap" }, '<span class="risk-badge ' + riskClass(p) + '">Risco ' + RISK_LABEL[d.risk_level] + "</span>"));
      card.appendChild(h("p", { class: "res-verdict" }, "O modelo indica que o cliente deve <b>" + (d.churn_prediction ? "cancelar" : "permanecer") + "</b>."));
      card.appendChild(h("p", { class: "res-meta mono" }, "HTTP " + r.status + " · " + fmtMs(r.ms) + " · modelo v" + esc(d.model_version)));
      out.appendChild(card);

      // fatores (explicabilidade da Regressão Logística)
      var ex = ChurnEngine.explain(payload).slice(0, 6), maxAbs = Math.max.apply(null, ex.map(function (e) { return Math.abs(e.contribution); })) || 1;
      var fx = h("div", { class: "factors" }, "<h4>O que mais pesou nesta predição</h4>");
      ex.forEach(function (e) {
        var up = e.contribution > 0, w = Math.abs(e.contribution) / maxAbs * 50;
        var val = e.feature === "MonthlyCharges" || e.feature === "TotalCharges" ? "R$ " + num(e.value, 2) : e.feature === "tenure" ? e.value + " meses" : L(e.value);
        fx.appendChild(h("div", { class: "factor" },
          '<span class="f-name">' + esc(FEATURE_LABEL[e.feature]) + ' <em>' + esc(val) + "</em></span>" +
          '<span class="f-bar"><span class="f-fill ' + (up ? "up" : "down") + '" style="' + (up ? "left:50%" : "right:50%") + ";width:" + w.toFixed(1) + '%"></span></span>' +
          '<span class="f-dir ' + (up ? "up" : "down") + '">' + (up ? "↑ risco" : "↓ risco") + "</span>"));
      });
      fx.appendChild(h("p", { class: "fine" }, "Contribuição de cada variável no modelo, em relação a um cliente médio da base. Calculado no navegador a partir dos coeficientes."));
      out.appendChild(fx);

      var act = h("div", { class: "actions" }, "<h4>Ações de retenção sugeridas</h4><ul class=\"dots\">" +
        retentionActions(payload).map(function (a) { return "<li>" + a + "</li>"; }).join("") + "</ul>");
      out.appendChild(act);

      var det = h("details", { class: "rest" });
      det.innerHTML = "<summary>Requisição e resposta REST</summary>" +
        '<p class="mono rest-line"><span class="method post">POST</span> /v1/predict</p>' +
        '<pre class="code sm"><code>' + jsonHtml(payload) + "</code></pre>" +
        '<p class="mono rest-line"><span class="status ' + statusClass(r.status) + '">' + r.status + "</span> resposta</p>" +
        '<pre class="code sm"><code>' + jsonHtml(d) + "</code></pre>" +
        '<p class="mono rest-line">cURL</p><pre class="code sm"><button class="copy" type="button">Copiar</button><code>' + esc(curlFor("POST", "/v1/predict", payload)) + "</code></pre>";
      out.appendChild(det);
      bindCopy(det);
    });
  }

  /* ---------- What-if ---------- */
  var WHATIF = {
    tenure: { title: "Meses como cliente", xs: range(0, 72, 3), key: "tenure", line: true, fmt: function (v) { return Math.round(v); } },
    MonthlyCharges: { title: "Mensalidade (R$)", xs: range(20, 120, 5), key: "MonthlyCharges", line: true, fmt: function (v) { return "R$ " + Math.round(v); } },
    Contract: { title: "Tipo de contrato", xs: SELECTS.Contract, key: "Contract" },
    PaymentMethod: { title: "Forma de pagamento", xs: SELECTS.PaymentMethod, key: "PaymentMethod" },
    TechSupport: { title: "Suporte técnico", xs: ["No", "Yes"], key: "TechSupport", needsNet: true }
  };
  function range(a, b, s) { var r = []; for (var i = a; i <= b; i += s) r.push(i); return r; }
  function initWhatIf() {
    $all("[data-whatif]").forEach(function (b) {
      b.addEventListener("click", function () {
        state.whatif = b.dataset.whatif;
        $all("[data-whatif]").forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
        renderWhatIf();
      });
    });
  }
  function renderWhatIf() {
    var cfg = WHATIF[state.whatif], base = state.profile, box = $("#whatifChart");
    $("#whatifProfile").innerHTML = "Perfil base: <b>" + esc(L(base.Contract)) + "</b>, " + base.tenure + " meses, R$ " + num(base.MonthlyCharges, 2) +
      "/mês, " + esc(L(base.InternetService)) + ", " + esc(L(base.PaymentMethod)) + '. <a href="#" data-goto="tab-predict">Alterar perfil</a>';
    var go = $("#whatifProfile a"); go.addEventListener("click", function (e) { e.preventDefault(); $("#" + go.dataset.goto).click(); });
    if (cfg.needsNet && base.InternetService === "No") {
      box.innerHTML = '<div class="alert">Este cliente não tem internet, então suporte técnico não se aplica. Escolha outra variável ou mude o perfil.</div>';
      $("#whatifStats").innerHTML = ""; return;
    }
    var scenarios = cfg.xs.map(function (x) {
      var s = Object.assign({}, base); s[cfg.key] = x;
      s.TotalCharges = Math.round(s.MonthlyCharges * s.tenure * 100) / 100;
      return s;
    });
    state.api.request("POST", "/v1/predict/batch", { customers: scenarios }).then(function (r) {
      if (!r.ok) { box.innerHTML = '<div class="alert err"><b>HTTP ' + (r.status || "—") + "</b> · " + esc(r.error) + "</div>"; $("#whatifStats").innerHTML = ""; return; }
      var probs = r.data.predictions.map(function (p) { return p.churn_probability; });
      box.innerHTML = "";
      var W = fitW(box, 720);
      if (cfg.line) box.appendChild(LineChart(cfg.xs, probs, cfg.title, base[cfg.key], cfg.fmt, W));
      else box.appendChild(BarChart(cfg.xs.map(function (x) { return L(x) + (x === base[cfg.key] ? " (atual)" : ""); }), probs.map(function (p) { return p * 100; }),
        { w: W, max: 100, bands: true, fmtY: function (v) { return v + "%"; }, fmtV: function (v) { return num(v, 1) + "%"; },
          cls: function (i) { return riskClass(probs[i]); }, label: "Risco por " + cfg.title }));
      var bi = probs.indexOf(Math.min.apply(null, probs)), wi = probs.indexOf(Math.max.apply(null, probs));
      var fx = cfg.line ? function (x) { return cfg.fmt(x) + (cfg.key === "tenure" ? " meses" : ""); } : L;
      $("#whatifStats").innerHTML =
        '<div class="kpi"><div class="l">Menor risco</div><div class="v">' + pct(probs[bi]) + '</div><div class="d">' + esc(fx(cfg.xs[bi])) + "</div></div>" +
        '<div class="kpi"><div class="l">Maior risco</div><div class="v">' + pct(probs[wi]) + '</div><div class="d">' + esc(fx(cfg.xs[wi])) + "</div></div>" +
        '<div class="kpi"><div class="l">Amplitude</div><div class="v">' + num((probs[wi] - probs[bi]) * 100, 1) + ' p.p.</div><div class="d">' + scenarios.length + " cenários em 1 requisição · " + fmtMs(r.ms) + "</div></div>";
    });
  }

  /* ---------- Análise em lote ---------- */
  function getSample() { if (!state.sample) state.sample = parseCSV(global.SAMPLE_CSV); return state.sample; }
  function initBatch() {
    $("#useSample").addEventListener("click", function () { runBatch(getSample(), "clientes_amostra.csv"); });
    $("#csvFile").addEventListener("change", function (e) {
      var file = e.target.files[0]; if (!file) return;
      var rd = new FileReader();
      rd.onload = function () {
        try { runBatch(parseCSV(String(rd.result)), file.name); }
        catch (err) { batchError("Não foi possível ler o CSV: " + err.message); }
      };
      rd.readAsText(file, "utf-8");
    });
    $("#batchFilters").addEventListener("change", renderBatchTable);
    $("#batchDownload").addEventListener("click", downloadBatch);
  }
  function batchError(msg) { $("#batchOut").hidden = true; $("#batchEmpty").hidden = false; $("#batchMsg").innerHTML = '<div class="alert err">' + msg + "</div>"; }
  function runBatch(csv, name) {
    var missing = FEATURES.filter(function (f) { return csv.columns.indexOf(f) === -1; });
    if (missing.length) return batchError("Colunas ausentes no CSV: <b>" + missing.map(esc).join(", ") + "</b>");
    var payloads = csv.rows.map(toPayload), bar = $("#batchProgress");
    $("#batchMsg").innerHTML = ""; bar.hidden = false; bar.value = 0;
    predictMany(state.api, payloads, function (p) { bar.value = p; }).then(function (r) {
      bar.hidden = true;
      if (!r.ok) return batchError("<b>HTTP " + (r.status || "—") + "</b> · " + esc(r.error));
      state.batch = { name: name, rows: csv.rows.map(function (row, i) { return Object.assign({}, row, r.data.predictions[i]); }), ms: r.ms, requests: r.requests, columns: csv.columns };
      renderBatch();
    });
  }
  function renderBatch() {
    var b = state.batch, rows = b.rows, n = rows.length;
    $("#batchEmpty").hidden = true; $("#batchOut").hidden = false;
    var churn = rows.filter(function (r) { return r.churn_prediction; }).length;
    var mean = rows.reduce(function (a, r) { return a + r.churn_probability; }, 0) / n;
    var high = rows.filter(function (r) { return r.risk_level === "alto"; }).length;
    $("#batchSummary").innerHTML = '<div class="alert ok"><b>' + num(n) + " clientes</b> analisados em " + fmtMs(b.ms) + " (" + b.requests + " requisiç" + (b.requests > 1 ? "ões" : "ão") + " POST /v1/predict/batch) · " + esc(b.name) + "</div>";
    $("#batchKpis").innerHTML =
      '<div class="kpi"><div class="l">Clientes</div><div class="v">' + num(n) + "</div></div>" +
      '<div class="kpi"><div class="l">Previstos para churn</div><div class="v">' + num(churn) + ' <small>(' + pct(churn / n, 0) + ")</small></div></div>" +
      '<div class="kpi"><div class="l">Risco médio</div><div class="v">' + pct(mean) + "</div></div>" +
      '<div class="kpi hl"><div class="l">Risco alto</div><div class="v">' + num(high) + "</div></div>";
    var real = $("#batchReal");
    if (b.columns.indexOf("Churn") !== -1) {
      var tp = 0, fn = 0, fp = 0, ok = 0;
      rows.forEach(function (r) {
        var y = /^(yes|1|true)$/i.test(r.Churn), yh = r.churn_prediction;
        if (y === yh) ok++; if (y && yh) tp++; if (y && !yh) fn++; if (!y && yh) fp++;
      });
      real.hidden = false;
      real.innerHTML = "O CSV traz o churn real: acurácia <b>" + pct(ok / n) + "</b> · recall <b>" + pct(tp / Math.max(1, tp + fn)) + "</b> · precisão <b>" + pct(tp / Math.max(1, tp + fp)) + "</b> nesta amostra.";
    } else real.hidden = true;

    var probs = rows.map(function (r) { return r.churn_probability; });
    var hist = $("#batchHist"); hist.innerHTML = ""; hist.appendChild(Histogram(probs, fitW(hist, 720)));
    var counts = { baixo: 0, medio: 0, alto: 0 }; rows.forEach(function (r) { counts[r.risk_level]++; });
    var dn = $("#batchDonut"); dn.innerHTML = ""; dn.appendChild(Donut(counts));
    dn.appendChild(h("div", { class: "legend" }, ["baixo", "medio", "alto"].map(function (k) {
      return '<span><i class="sw r-' + k + '"></i>' + RISK_LABEL[k] + " · " + counts[k] + "</span>";
    }).join("")));
    ["Contract", "PaymentMethod"].forEach(function (col) {
      var g = {}; rows.forEach(function (r) { (g[r[col]] = g[r[col]] || []).push(r.churn_probability); });
      var keys = Object.keys(g).sort(function (a, c) { return avg(g[c]) - avg(g[a]); });
      var vals = keys.map(function (k) { return avg(g[k]) * 100; });
      var box = $(col === "Contract" ? "#batchByContract" : "#batchByPay"); box.innerHTML = "";
      box.appendChild(BarChart(keys.map(L), vals, { max: niceCeil(Math.max(10, Math.max.apply(null, vals) * 1.15)), w: fitW(box, 460), h: 290, fmtY: function (v) { return Math.round(v) + "%"; },
        fmtV: function (v) { return num(v, 1) + "%"; }, cls: function (i) { return riskClass(vals[i] / 100); }, label: "Risco médio por " + col }));
    });
    var sel = $("#fContract"); sel.innerHTML = '<option value="">Todos os contratos</option>' +
      SELECTS.Contract.map(function (c) { return '<option value="' + esc(c) + '">' + esc(L(c)) + "</option>"; }).join("");
    renderBatchTable();
  }
  function avg(a) { return a.reduce(function (x, y) { return x + y; }, 0) / a.length; }
  function renderBatchTable() {
    if (!state.batch) return;
    var risks = $all('#batchFilters input[type="checkbox"]:checked').map(function (c) { return c.value; });
    var contract = $("#fContract").value, limit = 50;
    var view = state.batch.rows.filter(function (r) { return risks.indexOf(r.risk_level) !== -1 && (!contract || r.Contract === contract); })
      .sort(function (a, b) { return b.churn_probability - a.churn_probability; });
    var hasId = state.batch.columns.indexOf("customerID") !== -1, hasReal = state.batch.columns.indexOf("Churn") !== -1;
    var head = "<tr>" + (hasId ? "<th>Cliente</th>" : "<th>#</th>") + "<th>Prob. churn</th><th>Risco</th><th>Contrato</th><th>Meses</th><th>Mensalidade</th><th>Pagamento</th>" + (hasReal ? "<th>Churn real</th>" : "") + "</tr>";
    var body = view.slice(0, limit).map(function (r, i) {
      return "<tr><td class=\"mono\">" + esc(hasId ? r.customerID : i + 1) + '</td><td><div class="pbar"><span class="' + riskClass(r.churn_probability) + '" style="width:' + (r.churn_probability * 100).toFixed(1) + '%"></span></div>' + pct(r.churn_probability) +
        '</td><td><span class="pill ' + riskClass(r.churn_probability) + '">' + RISK_LABEL[r.risk_level] + "</span></td><td>" + esc(L(r.Contract)) + "</td><td>" + esc(r.tenure) + "</td><td>R$ " + num(parseFloat(r.MonthlyCharges), 2) + "</td><td>" + esc(L(r.PaymentMethod)) + "</td>" +
        (hasReal ? "<td>" + esc(L(r.Churn)) + "</td>" : "") + "</tr>";
    }).join("");
    $("#batchTable").innerHTML = "<thead>" + head + "</thead><tbody>" + (body || '<tr><td colspan="8" class="empty">Nenhum cliente com esses filtros.</td></tr>') + "</tbody>";
    $("#batchCount").textContent = view.length > limit ? "Mostrando os " + limit + " de maior risco, de " + num(view.length) + " clientes filtrados." : num(view.length) + " clientes filtrados.";
  }
  function downloadBatch() {
    if (!state.batch) return;
    var cols = state.batch.columns.concat(["churn_prediction", "churn_probability", "risk_level", "model_version"]);
    var lines = [cols.join(",")].concat(state.batch.rows.map(function (r) {
      return cols.map(function (c) { var v = r[c] === undefined ? "" : String(r[c]); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }).join(",");
    }));
    var blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" }), a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "predicoes_churn.csv"; document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }

  /* ---------- Laboratório da API ---------- */
  function without(o, k) { var c = Object.assign({}, o); delete c[k]; return c; }
  var LAB = [
    { name: "Cliente válido", method: "POST", path: "/v1/predict", body: DEFAULT_PROFILE, expect: 200 },
    { name: "Texto em campo numérico (tenure = \"dez\")", method: "POST", path: "/v1/predict", body: Object.assign({}, DEFAULT_PROFILE, { tenure: "dez" }), expect: 422 },
    { name: "Valor fora do limite (tenure = -5)", method: "POST", path: "/v1/predict", body: Object.assign({}, DEFAULT_PROFILE, { tenure: -5 }), expect: 422 },
    { name: "Categoria inexistente (Contract = \"Weekly\")", method: "POST", path: "/v1/predict", body: Object.assign({}, DEFAULT_PROFILE, { Contract: "Weekly" }), expect: 422 },
    { name: "Campo obrigatório ausente (sem MonthlyCharges)", method: "POST", path: "/v1/predict", body: without(DEFAULT_PROFILE, "MonthlyCharges"), expect: 422 },
    { name: "Campo extra não permitido", method: "POST", path: "/v1/predict", body: Object.assign({}, DEFAULT_PROFILE, { hacker: 1 }), expect: 422 },
    { name: "Regra de negócio: sem internet, com streaming", method: "POST", path: "/v1/predict", body: Object.assign({}, DEFAULT_PROFILE, { InternetService: "No" }), expect: 400 },
    { name: "Lote com 2 clientes", method: "POST", path: "/v1/predict/batch", body: { customers: [DEFAULT_PROFILE, LOYAL_PROFILE] }, expect: 200 },
    { name: "Lote vazio", method: "POST", path: "/v1/predict/batch", body: { customers: [] }, expect: 422 },
    { name: "Health check", method: "GET", path: "/health", body: null, expect: 200 },
    { name: "Métricas do modelo", method: "GET", path: "/v1/model/info", body: null, expect: 200 },
    { name: "Rota inexistente", method: "GET", path: "/v2/predict", body: null, expect: 404 }
  ];
  function initLab() {
    var preset = $("#labPreset");
    preset.innerHTML = LAB.map(function (s, i) { return '<option value="' + i + '">' + esc(s.name) + " · " + s.expect + "</option>"; }).join("");
    function load(i) {
      var s = LAB[i];
      $("#labMethod").value = s.method; $("#labPath").value = s.path;
      $("#labBody").value = s.body ? JSON.stringify(s.body, null, 2) : "";
      updateLabBody();
    }
    preset.addEventListener("change", function () { load(+preset.value); });
    $("#labMethod").addEventListener("change", updateLabBody);
    $("#labSend").addEventListener("click", sendLab);
    $("#labForm").addEventListener("submit", function (e) { e.preventDefault(); sendLab(); });
    $("#labFormat").addEventListener("click", function () {
      try { $("#labBody").value = JSON.stringify(JSON.parse($("#labBody").value), null, 2); } catch (e) { flash($("#labFormat"), "JSON inválido"); }
    });
    $("#labRunAll").addEventListener("click", runAllLab);
    load(0);
  }
  function flash(btn, t) { var o = btn.textContent; btn.textContent = t; setTimeout(function () { btn.textContent = o; }, 1400); }
  function updateLabBody() { $("#labBodyWrap").hidden = $("#labMethod").value === "GET"; }
  function sendLab() {
    var method = $("#labMethod").value, path = $("#labPath").value.trim() || "/", body = method === "GET" ? undefined : $("#labBody").value;
    var out = $("#labResponse"); out.classList.add("loading");
    state.api.request(method, path, body).then(function (r) {
      out.classList.remove("loading");
      out.innerHTML = '<div class="resp-head"><span class="status big ' + statusClass(r.status) + '">' + (r.status || "ERRO") + " " + esc(statusText(r.status)) + '</span><span class="mono muted">' + fmtMs(r.ms) + " · " + esc(state.api === LocalApi ? "navegador" : "API real") + "</span></div>" +
        (r.data !== null ? '<pre class="code sm"><code>' + jsonHtml(r.data) + "</code></pre>" : '<div class="alert err">' + esc(r.error) + "</div>") +
        '<p class="mono rest-line">cURL equivalente</p><pre class="code sm"><button class="copy" type="button">Copiar</button><code>' + esc(curlFor(method, path, body)) + "</code></pre>";
      bindCopy(out);
    });
  }
  function statusText(s) {
    return { 200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed", 422: "Unprocessable Entity", 500: "Internal Server Error", 503: "Service Unavailable" }[s] || "";
  }
  function runAllLab() {
    var tb = $("#labTable"), results = [];
    tb.innerHTML = '<tbody><tr><td class="empty">Executando…</td></tr></tbody>';
    var chain = Promise.resolve();
    LAB.forEach(function (s) {
      chain = chain.then(function () {
        return state.api.request(s.method, s.path, s.body === null ? undefined : s.body).then(function (r) { results.push({ s: s, r: r }); });
      });
    });
    chain.then(function () {
      var pass = results.filter(function (x) { return x.r.status === x.s.expect; }).length;
      $("#labScore").innerHTML = '<div class="kpi"><div class="l">Cenários</div><div class="v">' + results.length + '</div></div><div class="kpi hl"><div class="l">Aprovados</div><div class="v">' + pass + "/" + results.length +
        '</div></div><div class="kpi"><div class="l">Latência média</div><div class="v">' + fmtMs(avg(results.map(function (x) { return x.r.ms; }))) + "</div></div>";
      tb.innerHTML = "<thead><tr><th>Cenário</th><th>Rota</th><th>Esperado</th><th>Recebido</th><th></th><th>Resposta</th></tr></thead><tbody>" +
        results.map(function (x) {
          var ok = x.r.status === x.s.expect;
          return "<tr><td>" + esc(x.s.name) + '</td><td class="mono">' + x.s.method + " " + esc(x.s.path) + '</td><td class="mono">' + x.s.expect + '</td><td><span class="status ' + statusClass(x.r.status) + '">' + (x.r.status || "—") + "</span></td><td>" + (ok ? '<span class="ok-mark" aria-label="aprovado">✓</span>' : '<span class="bad-mark" aria-label="reprovado">✗</span>') +
            '</td><td class="resp-cell">' + esc(x.r.ok ? "sucesso" : x.r.error) + "</td></tr>";
        }).join("") + "</tbody>";
    });
  }

  function refreshAll() {
    runPredict();
    if (!$("#panel-whatif").hidden) renderWhatIf();
    if (state.batch) $("#batchMsg").innerHTML = '<div class="alert">Motor alterado. Rode a análise de novo para usar ' + esc(state.api === LocalApi ? "o modelo no navegador" : "a API real") + ".</div>";
  }

  /* ===================================================================
     6. PÁGINA
     =================================================================== */
  function bindCopy(root) {
    $all(".copy", root).forEach(function (b) {
      if (b.dataset.bound) return; b.dataset.bound = "1";
      b.addEventListener("click", function () {
        var t = b.parentElement.querySelector("code").innerText;
        var done = function () { flash(b, "Copiado!"); };
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(done, function () { flash(b, "Selecione e copie"); });
        else flash(b, "Selecione e copie");
      });
    });
  }
  function initPage() {
    var root = document.documentElement, btn = $("#themeBtn");
    try { var saved = localStorage.getItem("theme"); if (saved) root.setAttribute("data-theme", saved); } catch (e) {}
    btn.addEventListener("click", function () {
      var dark = root.getAttribute("data-theme") === "dark" || (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
      var next = dark ? "light" : "dark"; root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
    });

    var shots = [
      ["visao", "Visão geral", "saúde da API e métricas do modelo", "GET /health · GET /v1/model/info"],
      ["predicao", "Predição individual", "velocímetro de risco e ações de retenção", "POST /v1/predict"],
      ["whatif", "Simulador what-if", "como o risco muda ao trocar o tipo de contrato", "POST /v1/predict/batch"],
      ["lote", "Análise em lote", "300 clientes em 3 requisições, com ranking de prioridade", "POST /v1/predict/batch"],
      ["lab", "Laboratório da API", "9 de 9 cenários com o status esperado", "200 · 400 · 422"]
    ];
    var img = $("#shotImg"), tabs = $all(".tab");
    function show(i) {
      img.src = "assets/" + shots[i][0] + ".webp"; img.alt = "Dashboard Streamlit: " + shots[i][1];
      $("#shotCap").innerHTML = "<b>" + shots[i][1] + ":</b> " + shots[i][2]; $("#shotEp").textContent = shots[i][3];
      tabs.forEach(function (t, j) { t.setAttribute("aria-selected", j === i ? "true" : "false"); });
    }
    tabs.forEach(function (t) { t.addEventListener("click", function () { show(+t.dataset.i); }); });
    if (img) show(0);

    var lb = $("#lightbox"), lbi = $("#lightboxImg");
    $all(".shot img, .hero-shot img").forEach(function (e) {
      e.addEventListener("click", function () { lbi.src = e.src; lbi.alt = e.alt; lb.classList.add("open"); });
    });
    lb.addEventListener("click", function () { lb.classList.remove("open"); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") lb.classList.remove("open"); });

    bindCopy(document);

    $all("[data-open-tab]").forEach(function (a) {
      a.addEventListener("click", function () { var t = $("#" + a.dataset.openTab); if (t) t.click(); });
    });

    var els = $all(".fade");
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }); }, { threshold: 0.06 });
      els.forEach(function (e) { io.observe(e); });
    } else els.forEach(function (e) { e.classList.add("in"); });
  }

  function boot() {
    initPage();
    if (!global.CHURN_MODEL) { $("#demoApp").innerHTML = '<div class="alert err">Não foi possível carregar assets/model.js.</div>'; return; }
    initTabs(); initForm(); initWhatIf(); initBatch(); initLab(); initEngine();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})(typeof window !== "undefined" ? window : globalThis);
