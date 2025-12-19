tour spec

scope

add a live-computed “tour mode” at route /tour that teaches 7 ideas using the real model + real tensors, with causal ablations and pretrain-vs-sft A/B, while reusing the existing chat stack.

hard constraints
	•	live-computed (no prerecorded traces)
	•	prompts are prefilled but editable
	•	transcript/generation panel is on the right (reuses existing chat UI patterns)
	•	reuse existing SSE streaming endpoint (currently POST /chat/stream)
	•	no substring stop hacks; stop is token-native (<|eot|>)
	•	ablations are inference-only (no training changes)

⸻

layout and UX invariants

/tour page layout

three columns:
	1.	left: chapters

	•	list of 7 chapters (fixed ordering)
	•	clicking a chapter loads its defaults + prefilled prompt (but user can edit)
	•	a “reset chapter” button restores defaults + prompt

	2.	center: chapter visualization panel

	•	exactly one primary visualization per chapter
	•	exactly one primary control per chapter
	•	“advanced” accordion may expose extra knobs, but must be collapsed by default

	3.	right: transcript + live generation panel

	•	same visual language/components as your existing chat page
	•	shows:
	•	current prompt/messages
	•	streaming generated output for run A and (when applicable) run B
	•	replay controls when a run completes (scrub per-step)

primary controls per chapter (locked)
	1.	tokens are the interface → toggle “show ids”
	2.	guesses one token at a time → step scrubber
	3.	decoding changes behavior → decoding preset dropdown
	4.	attention is retrieval → head dropdown (layer fixed default)
	5.	prove attention matters → ablation toggle baseline vs disable selected head
	6.	layers build representations → layer dropdown (head fixed default)
	7.	sft makes it an assistant → checkpoint dropdown (pretrain vs sft)

⸻

chapter program contract

store as a static config file in the frontend: tour_chapters.json (or .ts exporting JSON-equivalent). this file is the tour program.

TourChapter schema

type CheckpointId = "pretrain" | "sft";

type DecodingPresetId =
  | "greedy"
  | "topk_default"
  | "topp_default";

type ChapterMode =
  | "tokens"
  | "distribution"
  | "sampling_fork"
  | "attention"
  | "ablation_head"
  | "layers"
  | "sft_ab";

type PrimaryControl =
  | "toggle_show_ids"
  | "scrub_step"
  | "select_decoding_preset"
  | "select_head"
  | "toggle_ablation"
  | "select_layer"
  | "select_checkpoint";

type TourChapter = {
  id: string;                 // stable slug, e.g. "tokens"
  title: string;              // short title
  goal: string;               // 1 sentence
  prefill_prompt: string;     // editable by user
  mode: ChapterMode;

  defaults: {
    checkpoint_id: CheckpointId;
    decoding_preset_id: DecodingPresetId;
    max_new_tokens: number;     // capped server-side too
    seed: number;               // fixed default; user can reroll
    trace: {
      kind: "none" | "attn_row_topn";
      layer: number | null;     // required if kind != none
      head: number | null;      // optional; client can pick
      topn: number;             // 8 or 16 (locked default)
    };
    ablation_preset_id: string | null; // references an AblationPreset map
  };

  primary_control: PrimaryControl;
};

required chapter list (locked)
	•	tokens
	•	distribution
	•	sampling
	•	attention
	•	prove_attention
	•	layers
	•	sft

⸻

backend API contracts

1) GET /model/info

frontend uses this to render metadata, validate chapter defaults, and map checkpoint ids.

response:

{
  "model": {
    "name": "niels-gpt",
    "context_length": 256,
    "vocab_size": 8192,
    "special_tokens": {
      "sys": {"text":"<|sys|>", "id": 8001},
      "usr": {"text":"<|usr|>", "id": 8002},
      "asst":{"text":"<|asst|>","id":8003},
      "eot": {"text":"<|eot|>", "id": 8004}
    }
  },
  "checkpoints": [
    {"id":"pretrain","path":"...","step":20000,"sha256":"..."},
    {"id":"sft","path":"...","step":4000,"sha256":"..."}
  ],
  "limits": {
    "max_prompt_bytes": 16384,
    "max_new_tokens": 512,
    "max_concurrent_streams": 4
  }
}

2) POST /chat/stream (SSE)

tour reuses this endpoint. it must support:
	•	deterministic sampling via seed
	•	multiplexed A/B runs in one stream
	•	optional tracing per run
	•	optional ablation per run
	•	token-native stop on <|eot|>

request schema: ChatStreamRequestV2

type Role = "system" | "user" | "assistant";

type Message = { role: Role; content: string };

type DecodingParams = {
  max_new_tokens: number;
  temperature: number;      // 0 => greedy
  top_k?: number | null;
  top_p?: number | null;
  seed: number;             // required in tour; optional in normal chat
};

type TraceConfig =
  | { kind: "none" }
  | {
      kind: "attn_row_topn";
      layer: number;        // 0..L-1
      topn: number;         // 8 or 16
    };

type AblateConfig = {
  disable_heads?: Array<{ layer: number; head: number }>;
  disable_mlp_layers?: number[];
  disable_attn_layers?: number[];
  skip_layers?: number[];
  ban_token_ids?: number[]; // applied at decoding time
};

type RunSpec = {
  run_id: "A" | "B";
  checkpoint_id: "pretrain" | "sft";
  decoding: DecodingParams;
  trace: TraceConfig;
  ablate?: AblateConfig | null;
};

type ChatStreamRequestV2 = {
  mode: "chat" | "tour";        // tour forces extra validation
  messages: Message[];          // prompt is derived server-side
  runs: RunSpec[];              // 1 run for most chapters, 2 for A/B chapters
};

server-side validation invariants
	•	prompt bytes (utf-8) ≤ 16384 else 413 json error
	•	rate limits remain enforced (429)
	•	mode="tour":
	•	runs.length must be 1 or 2
	•	seed required
	•	max_new_tokens must be ≤ server max (e.g. 256 for tour, 512 for chat)
	•	if temperature == 0 then ignore top_k/top_p and do greedy
	•	if both top_k and top_p are set, apply top_k first, then top_p on the truncated set (locked precedence)

SSE event protocol (locked)
all events are event: <name>\ndata: <json>\n\n (standard SSE).

common fields for every event payload:
	•	run_id: "A" | "B"
	•	ts_ms: number (server timestamp)

events:
	1.	meta

{
  "run_id":"A",
  "model": {"checkpoint_id":"sft","context_length":256},
  "decoding": {"temperature":0.9,"top_k":50,"top_p":null,"seed":123,"max_new_tokens":128},
  "trace": {"kind":"attn_row_topn","layer":2,"topn":8},
  "ablate": {"disable_heads":[{"layer":2,"head":1}]}
}

	2.	step (emitted once per generated token, before the token)

{ "run_id":"A", "step": 17, "t_ctx": 241 }

	3.	topk

{
  "run_id":"A",
  "step":17,
  "entropy": 2.31,
  "candidates": [
    {"id": 1234, "text":" the", "logit": 4.12, "prob": 0.19},
    {"id": 987,  "text":" a",   "logit": 3.91, "prob": 0.15}
  ]
}

	4.	token

{
  "run_id":"A",
  "step":17,
  "token": {"id":1234, "text":" the"},
  "text_delta":" the"
}

	5.	attn_row_topn (optional, per step, if trace enabled)
stream sparse attention: top-n attended positions per head.

{
  "run_id":"A",
  "step":17,
  "layer":2,
  "topn":8,
  "heads": [
    {"pos":[240, 12, 44], "w":[0.41, 0.09, 0.07]},
    {"pos":[123, 240],    "w":[0.22, 0.18]}
  ]
}

invariants:
	•	heads.length == H
	•	positions are 0..t_ctx-1 (in current cropped window)
	•	weights are attention probabilities (sum of all probs is 1.0, but sparse list won’t sum to 1)

	6.	done

{
  "run_id":"A",
  "stop_reason":"eot" | "max_new_tokens" | "error",
  "tokens_generated": 128,
  "ms_total": 8421
}

	7.	error

{ "run_id":"A", "code":"RATE_LIMIT"|"BAD_REQUEST"|"INTERNAL", "message":"..." }

3) POST /inspect/full_attn

on-demand full attention for a (run, step, layer, head). tour keeps this behind “advanced”.

request:

{
  "checkpoint_id":"sft",
  "messages":[...],
  "decoding":{"seed":123,"temperature":0.0,"max_new_tokens":1},
  "step":17,
  "layer":2,
  "head":1
}

response (quantized):

{
  "shape":[t_ctx, t_ctx],
  "encoding":"u8_affine",
  "scale":0.0039,
  "zero_point":0,
  "values_u8": [0,0,1, ...]  // flattened row-major
}

hard limits:
	•	only allowed for t_ctx <= 256 (or your current T)
	•	rate limited separately if needed

⸻

inference determinism + safety invariants
	•	server sets model.eval() always for inference
	•	sampling RNG uses a cpu torch.Generator seeded with request seed
	•	A/B is computed sequentially per request and multiplexed into one SSE stream
	•	ablations are implemented via an explicit AblateConfig passed down the forward path used by inference (no global hooks)
	•	tour forces seed so A/B isn’t noise

⸻

caching invariants

implement an in-memory LRU cache for completed streams:
	•	key = stable hash of the request payload (mode + messages + runs)
	•	value = full list of SSE events (already serialized JSON payloads)
	•	on identical request, server replays cached events immediately
	•	cache size limit: e.g. 128 entries (configurable)

⸻

pr roadmap

design principle: each pr has (a) one clear deliverable, (b) minimal file surface, (c) tests that force correctness. every pr is one claude prompt.

pr-01 — tour contracts + model info endpoint

goal
	•	formalize contracts so frontend/backend cannot drift
	•	expose GET /model/info for UI bootstrap + defaults validation

scope
	•	add shared schemas/types:
	•	backend/schemas/tour.py (pydantic models for request/events)
	•	frontend/src/types/tour.ts (ts types mirroring the same)
	•	implement GET /model/info
	•	add checkpoint registry (source of truth):
	•	backend/checkpoints/registry.json mapping ids → path + metadata

files
	•	backend: add only new files + a single router registration file
	•	frontend: types only (no UI yet)

acceptance tests
	•	backend unit: /model/info returns both checkpoints and special tokens
	•	backend unit: schema validation rejects invalid checkpoint_id, invalid trace.layer, invalid runs.length in tour mode
	•	frontend typecheck passes with the new types

non-goals
	•	no changes to generation logic
	•	no UI

⸻

pr-02 — SSE protocol v2 + multiplex A/B + deterministic seeds

goal
	•	upgrade /chat/stream to accept ChatStreamRequestV2 with runs[]
	•	stream meta/step/topk/token/done/error with run_id
	•	enforce tour validation (seed required, token limits)

scope
	•	adapt existing SSE endpoint:
	•	request parsing → ChatStreamRequestV2
	•	sequentially execute runs A then B, multiplex events
	•	ensure token-native stop (<|eot|>)
	•	ensure determinism:
	•	cpu generator seeded per run

files
	•	backend:
	•	existing SSE endpoint file (surgical edit)
	•	new helper module: backend/streaming/sse.py (event emitter)
	•	new helper module: backend/infer/run.py (single-run execution returning event stream)

acceptance tests
	•	integration-ish test:
	•	send a request with runs=[A,B] and verify:
	•	first event is meta for A, then some token events, then done for A; then meta for B…
	•	every event has run_id
	•	determinism test:
	•	same request twice returns identical token sequence for run A (compare token.id list)
	•	stop test:
	•	if model emits <|eot|>, server stops and returns done.stop_reason=="eot"

non-goals
	•	no attention tracing yet
	•	no ablations yet
	•	no caching yet

⸻

pr-03 — streaming sparse attention rows + full attention inspect

goal
	•	enable trace.kind="attn_row_topn" streaming per step
	•	implement POST /inspect/full_attn quantized response

scope
	•	inference plumbing:
	•	add TraceConfig to inference call path
	•	capture attention probabilities for the configured layer
	•	per step, emit attn_row_topn with top-n positions per head
	•	full attention endpoint:
	•	compute full attention matrix for a requested (step, layer, head)
	•	quantize to u8 + scale

files
	•	backend:
	•	backend/infer/trace.py (trace extraction + top-n)
	•	minimal edits to model forward to optionally return attention probs for one layer without changing training paths
	•	requirement: inference path can request attention; training path remains unchanged
	•	backend/routes/inspect.py for full attn endpoint

acceptance tests
	•	trace test:
	•	when trace enabled, every generated step emits an attn_row_topn event
	•	heads.length == H and all pos are within [0, t_ctx)
	•	full attn test:
	•	response shape is [t_ctx,t_ctx]
	•	values are u8 list of length t_ctx*t_ctx
	•	quantization invariants hold (0<=u8<=255, scale>0)

non-goals
	•	no ablations yet
	•	no frontend

⸻

pr-04 — ablations (inference-only, no hooks) + token bans

goal
	•	make causal “prove it” possible:
	•	disable a head
	•	disable mlp in a layer
	•	disable attention in a layer
	•	skip a layer
	•	ban token ids during decoding

scope
	•	implement AblateConfig applied only in inference:
	•	thread-safe, request-scoped
	•	no global state, no hooks
	•	decoding-time bans:
	•	set logits for banned token ids to -inf before sampling

files
	•	backend:
	•	backend/infer/ablate.py
	•	minimal edits in model blocks to respect ablation config when provided (in inference path)

acceptance tests
	•	head ablation changes output:
	•	same seed + same prompt:
	•	run A baseline tokens != run B with disable_heads=[...] for at least one token within first N steps
	•	ban tokens works:
	•	ban <|usr|> and <|sys|> ids; verify they never appear in generated tokens

non-goals
	•	no frontend
	•	no refactors to training

⸻

pr-05 — caching + tour page (live) + chapter viz scaffolding

goal
	•	/tour exists, uses live /chat/stream, and renders all 7 chapters with the correct layout and primary controls
	•	server caches completed streams to make it usable

scope (backend)
	•	add LRU cache for completed streams keyed by request hash
	•	if cache hit: replay events immediately

scope (frontend)
	•	route /tour
	•	left chapter nav from tour_chapters
	•	center panel renders the chapter’s single viz:
	•	tokens view
	•	distribution view (topk + entropy)
	•	sampling fork view (calls two or three runs; can be multiplexed as A/B only at first—keep it simple)
	•	attention spotlight
	•	ablation compare
	•	layer sweep
	•	sft compare (checkpoint A/B)
	•	right panel reuses transcript/generation UI components
	•	global: prompt editable + reset button + reroll seed button

files
	•	frontend:
	•	frontend/src/tour/tour_chapters.ts
	•	frontend/src/pages/tour.tsx (or your router equivalent)
	•	frontend/src/tour/components/* (small, chapter-specific)
	•	reuse existing chat transcript component; only wrap it
	•	backend:
	•	backend/cache/lru.py + small integration into SSE route

acceptance tests
	•	frontend e2e-ish (playwright or minimal):
	•	page loads, shows 7 chapters
	•	clicking each chapter updates center panel mode and prefilled prompt
	•	backend cache test:
	•	same request twice: second run returns meta almost immediately and token stream matches cached sequence

non-goals
	•	no full attention matrix UI by default (advanced only)
	•	no polishing animations

⸻

pr-06 — A/B UX polish + “advanced” panel + model checkpoint downloader

goal
	•	make the tour resilient in the real world:
	•	ensure pretrain/sft checkpoints are available at startup (download if missing)
	•	add advanced panel toggles (full attn fetch, extra knobs)
	•	improve A/B side-by-side replay UI

scope
	•	backend: checkpoint bootstrap
	•	on startup: verify registry paths exist; if not, download from configured URL(s)
	•	expose failure clearly in /model/info or logs
	•	frontend:
	•	A/B compare visualization improvements
	•	“advanced” accordion for full attn matrix fetch + extra knobs

acceptance tests
	•	startup test:
	•	if checkpoint missing, downloader is invoked and server still starts (or fails with a precise error)
	•	UI test:
	•	advanced panel is hidden by default and toggles open

non-goals
	•	no new visualization types
	•	no performance micro-optimizations unless required

⸻

common PR prompt constraints (for claude)

include this block in every PR prompt pack:
	•	do not change training code paths, data pipelines, or model weights loading beyond what the PR explicitly lists
	•	do not introduce new frameworks
	•	do not refactor unrelated files
	•	add/adjust tests exactly as specified
	•	preserve existing API behavior for non-tour chat mode
	•	all new behavior must be behind explicit flags/fields (mode, trace, ablate, etc.)
