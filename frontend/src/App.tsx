import { useCallback, useEffect, useState } from "react";
import {
  ArrowRight,
  AudioLines,
  BookOpen,
  Check,
  ChevronRight,
  Cpu,
  Database,
  FileText,
  FlaskConical,
  Image,
  Layers3,
  LoaderCircle,
  MessageSquare,
  Moon,
  Search,
  Sparkles,
  Sun,
  Upload,
  Video,
  Zap,
} from "lucide-react";
import {
  Background,
  Controls,
  ReactFlow,
  type Node,
  type Edge,
} from "@xyflow/react";
import { ResponsiveContainer, BarChart, Bar, XAxis, Tooltip } from "recharts";
import "@xyflow/react/dist/style.css";
import "./App.css";

type Modality = "text" | "image" | "audio" | "video";
type Result = {
  id: string;
  file_name: string;
  modality: Modality;
  rank?: number;
  similarity?: number;
  text?: string;
  url: string;
  page?: number;
  start_time?: number;
  end_time?: number;
  vectors?: number;
  index_seconds?: number;
  explanation?: string;
  keyword_hits?: number;
};
type Job = {
  id: string;
  file_name: string;
  stage: string;
  error?: string;
  vectors?: number;
  seconds?: number;
};
type Status = {
  model: string;
  state: string;
  device: string | null;
  dimensions: number;
  parameter_count: number | null;
  warning?: string;
  error?: string;
};
type Stats = {
  files: number;
  vectors: number;
  modalities: Record<Modality, number>;
};
const icons = { text: FileText, image: Image, audio: AudioLines, video: Video };
const labels = {
  text: "Text",
  image: "Images",
  audio: "Audio",
  video: "Video",
};
const pages = [
  "Overview",
  "Upload anything",
  "Semantic search",
  "Evidence chat",
  "How it works",
];
const pageIcons = [Layers3, Upload, Search, MessageSquare, BookOpen];
const examples = [
  "Find information about worker safety.",
  "Find documents or media related to company vehicles.",
  "Find information about employee training.",
];
async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const body = await response
    .json()
    .catch(() => ({ detail: `Server returned ${response.status}` }));
  if (!response.ok)
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail),
    );
  return body;
}
function post<T>(path: string, body: unknown) {
  return api<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
function Preview({ result: r }: { result: Result }) {
  const fragment =
    r.start_time !== undefined
      ? `#t=${r.start_time},${r.end_time}`
      : r.page
        ? `#page=${r.page}`
        : "";
  if (r.modality === "image")
    return (
      <img className="preview" src={r.url} alt={r.file_name} loading="lazy" />
    );
  if (r.modality === "audio")
    return (
      <div className="audio-preview">
        <AudioLines size={40} />
        <audio controls preload="none" src={r.url + fragment} />
      </div>
    );
  if (r.modality === "video")
    return (
      <video
        className="preview"
        controls
        preload="metadata"
        src={r.url + fragment}
      />
    );
  return (
    <div className="text-preview">
      <FileText size={24} />
      <p>
        {r.text?.slice(0, 260) ||
          "Open the original document to explore its contents."}
      </p>
    </div>
  );
}
function ResultCard({ result: r }: { result: Result }) {
  const Icon = icons[r.modality];
  return (
    <article className="result-card">
      <Preview result={r} />
      <div className="result-content">
        <div className="row between">
          <span className={`tag ${r.modality}`}>
            <Icon size={13} />
            {labels[r.modality]}
          </span>
          {r.rank && <small>#{r.rank}</small>}
        </div>
        <a
          href={
            r.url +
            (r.page
              ? `#page=${r.page}`
              : r.start_time !== undefined
                ? `#t=${r.start_time}`
                : "")
          }
          target="_blank"
          rel="noreferrer"
        >
          <h3>{r.file_name}</h3>
        </a>
        <small>
          {r.page
            ? `Page ${r.page}`
            : r.start_time !== undefined
              ? `${r.start_time.toFixed(1)}–${r.end_time?.toFixed(1)} seconds`
              : "Original source"}
        </small>
        {r.similarity !== undefined && (
          <div className="score">
            <div className="row between">
              <span>Cosine similarity</span>
              <strong>{r.similarity.toFixed(3)}</strong>
            </div>
            <div className="meter">
              <i style={{ width: `${Math.max(0, r.similarity) * 100}%` }} />
            </div>
            <small>{r.explanation}</small>
          </div>
        )}
        {r.vectors !== undefined && (
          <p>
            {r.vectors} vectors · {r.index_seconds?.toFixed(2)}s indexing
          </p>
        )}
      </div>
    </article>
  );
}
function Diagram() {
  const nodes: Node[] = [
    ...(["text", "image", "audio", "video"] as Modality[]).map((m, i) => ({
      id: m,
      position: { x: 10, y: i * 65 },
      data: { label: labels[m] },
    })),
    {
      id: "model",
      position: { x: 235, y: 90 },
      data: { label: "EmbeddingGemma 2\nLocal multimodal encoder" },
      className: "flow-model",
    },
    {
      id: "store",
      position: { x: 485, y: 90 },
      data: { label: "One unified database\n768 dimensions · cosine" },
      className: "flow-store",
    },
  ];
  const edges: Edge[] = [
    ...["text", "image", "audio", "video"].map((m) => ({
      id: m,
      source: m,
      target: "model",
      animated: true,
    })),
    { id: "out", source: "model", target: "store", animated: true },
  ];
  return (
    <div className="flow">
      <ReactFlow nodes={nodes} edges={edges} fitView nodesConnectable={false}>
        <Background gap={22} color="#dedee7" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
function Heading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="page-heading">
      <span className="eyebrow">{eyebrow}</span>
      <h1>{title}</h1>
      <p>{description}</p>
    </div>
  );
}
function Empty({ title, text }: { title: string; text: string }) {
  return (
    <div className="empty">
      <div className="empty-icon">
        <Layers3 size={28} />
      </div>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState(0),
    [dark, setDark] = useState(false);
  const [status, setStatus] = useState<Status | null>(null),
    [stats, setStats] = useState<Stats | null>(null);
  const [files, setFiles] = useState<Result[]>([]),
    [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [query, setQuery] = useState(""),
    [queryType, setQueryType] = useState<"text" | "image" | "audio">("text");
  const [queryFile, setQueryFile] = useState<File | null>(null),
    [filter, setFilter] = useState<Modality | "">("");
  const [results, setResults] = useState<Result[]>([]),
    [keywords, setKeywords] = useState<Result[]>([]);
  const [searched, setSearched] = useState(false),
    [seconds, setSeconds] = useState<number | null>(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<
    {
      question: string;
      answer: string;
      sources: Result[];
      stages: Record<string, unknown>[];
    }[]
  >([]);
  const [step, setStep] = useState(0);
  const [texts, setTexts] = useState([
    "An employee applied for annual leave.",
    "A worker requested vacation.",
    "The server temperature is increasing.",
  ]);
  const [matrix, setMatrix] = useState<number[][] | null>(null);
  const refresh = useCallback(async () => {
    try {
      const [s, t, f] = await Promise.all([
        api<Status>("/api/model/status"),
        api<Stats>("/api/stats"),
        api<Result[]>("/api/files"),
      ]);
      setStatus(s);
      setStats(t);
      setFiles(f);
    } catch (e) {
      setError(String(e));
    }
  }, []);
  useEffect(() => {
    const initial = setTimeout(() => void refresh(), 0);
    const id = setInterval(() => void refresh(), 10000);
    return () => {
      clearTimeout(initial);
      clearInterval(id);
    };
  }, [refresh]);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  }, [dark]);
  useEffect(() => {
    const active = jobs.filter(
      (j) => !["Searchable", "Error"].includes(j.stage),
    );
    if (!active.length) return;
    const id = setInterval(async () => {
      try {
        const updates = await Promise.all(
          active.map((j) => api<Job>(`/api/jobs/${j.id}`)),
        );
        setJobs((current) =>
          current.map((j) => updates.find((u) => u.id === j.id) || j),
        );
        if (updates.some((j) => j.stage === "Searchable")) void refresh();
      } catch (e) {
        setError(String(e));
      }
    }, 1200);
    return () => clearInterval(id);
  }, [jobs, refresh]);
  async function action(task: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await task();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  async function upload(incoming: FileList | File[]) {
    await action(async () => {
      for (const file of Array.from(incoming)) {
        const data = new FormData();
        data.append("file", file);
        const r = await api<{ job_id: string }>("/api/upload", {
          method: "POST",
          body: data,
        });
        setJobs((current) => [
          ...current,
          { id: r.job_id, file_name: file.name, stage: "Upload" },
        ]);
      }
    });
  }
  async function loadDemo() {
    await action(async () => {
      const r = await post<{ jobs: { job_id: string }[] }>(
        "/api/demo/load",
        {},
      );
      const loaded = await Promise.all(
        r.jobs.map((j) => api<Job>(`/api/jobs/${j.job_id}`)),
      );
      setJobs((current) => [...current, ...loaded]);
      setPage(1);
    });
  }
  async function search() {
    await action(async () => {
      let r: { results: Result[]; keyword_results: Result[]; seconds: number };
      if (queryType === "text") {
        if (!query.trim()) throw new Error("Enter a question.");
        r = await post("/api/search/text", {
          text: query,
          modality: filter || null,
        });
      } else {
        if (!queryFile) throw new Error(`Choose an ${queryType} query file.`);
        const data = new FormData();
        data.append("file", queryFile);
        if (filter) data.append("modality", filter);
        r = await api(`/api/search/${queryType}`, {
          method: "POST",
          body: data,
        });
      }
      setResults(r.results);
      setKeywords(r.keyword_results);
      setSeconds(r.seconds);
      setSearched(true);
    });
  }
  const learning = [
    [
      "Multimodal inputs",
      "Text and code, photographs, spoken audio, and video each contain different signals. Native encoders process these inputs; captions and transcripts do not replace their embeddings.",
    ],
    [
      "Local encoding",
      "The full local model includes text, vision, and audio encoders. Every input becomes a normalized 768-dimensional vector. Text uses task-specific prompts; media enters its native pipeline.",
    ],
    [
      "Unified vector database",
      "All vectors coexist in one persistent Chroma collection. Cosine similarity compares directions. Original files stay on disk, linked by stable content hashes and page or clip references.",
    ],
    [
      "Cross-modal retrieval",
      "A text, image, or audio query enters the same embedding space. Nearby vectors become ranked results across all modalities. A similarity score measures alignment; it is not a probability.",
    ],
    [
      "Agentic RAG",
      "LangGraph retrieves evidence, checks for readable passages, and optionally rewrites the query once. OpenRouter receives those passages to produce citations. A text model cannot read an embedding as media evidence.",
    ],
  ];
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage(0);
          }}
        >
          <div className="brand-mark">
            <Layers3 size={25} />
          </div>
          <span>
            one<span className="brand-light">space</span>
            <small>EMBEDDINGGEMMA 2 LAB</small>
          </span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {pages.map((name, i) => {
            const Icon = pageIcons[i];
            return (
              <button
                aria-label={name}
                className={page === i ? "nav-item active" : "nav-item"}
                onClick={() => {
                  setPage(i);
                  setError("");
                }}
                key={name}
              >
                <Icon size={18} />
                {name}
                {page === i && <ChevronRight size={14} className="nav-arrow" />}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-card">
            <span className="status-dot" />
            <strong>Local-first by design</strong>
            <p>Your embeddings and files stay on this machine.</p>
            <small>Chat excerpts → OpenRouter</small>
          </div>
          <button className="theme-button" onClick={() => setDark(!dark)}>
            {dark ? <Sun size={16} /> : <Moon size={16} />}
            {dark ? "Light appearance" : "Dark appearance"}
          </button>
          <div className="sidebar-footer">
            A shared space for every idea.
            <br />
            Google model · Independent demo
          </div>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={13} />
            <strong>{pages[page]}</strong>
          </div>
          <div className="row">
            <span className="version">RESEARCH DEMO</span>
            <span
              className={`connection ${status?.state === "ready" ? "ready" : ""}`}
            >
              <span className="status-dot" />
              {status ? status.state.replace("_", " ") : "Connecting"}
            </span>
          </div>
        </header>
        <div className="page-content">
          {error && (
            <div role="alert" className="error">
              <strong>Something needs attention</strong>
              <p>{error}</p>
              <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
          {page === 0 && (
            <>
              <div className="eyebrow">
                <Sparkles size={14} /> THE MULTIMODAL KNOWLEDGE LAB
              </div>
              <div className="hero">
                <div>
                  <h1>
                    One AI model.
                    <br />
                    Every type of <span>knowledge.</span>
                  </h1>
                  <p>
                    Turn documents, images, audio, and video into one searchable
                    space. Discover connections beyond file types.
                  </p>
                  <div className="row hero-actions">
                    <button className="primary" onClick={() => setPage(2)}>
                      <Search size={17} />
                      Explore semantic search
                      <ArrowRight size={16} />
                    </button>
                    <button
                      className="secondary"
                      onClick={() => void loadDemo()}
                      disabled={busy}
                    >
                      Load demo dataset
                    </button>
                  </div>
                  <div className="hero-note">
                    <span className="status-dot" />
                    Local inference · Native multimodal embeddings · Open model
                  </div>
                </div>
                <div
                  className="orbit-art"
                  aria-label="Four modalities sharing one vector space"
                >
                  <div className="orbit o1" />
                  <div className="orbit o2" />
                  <div className="orbit o3" />
                  <div className="orbit-center">
                    <Layers3 size={42} />
                    <strong>ONE SPACE</strong>
                    <small>768 dimensions</small>
                  </div>
                  {(["text", "image", "audio", "video"] as Modality[]).map(
                    (m, i) => {
                      const Icon = icons[m];
                      return (
                        <div key={m} className={`orbit-item item-${i} ${m}`}>
                          <Icon size={23} />
                          <span>{labels[m]}</span>
                        </div>
                      );
                    },
                  )}
                  <span className="art-caption">
                    DIFFERENT INPUTS. SHARED MEANING.
                  </span>
                </div>
              </div>
              <div className="stats-grid">
                {[
                  {
                    label: "Embedding model",
                    value: "EmbeddingGemma 2",
                    note: status?.model || "Awaiting backend",
                    icon: Cpu,
                  },
                  {
                    label: "Vector dimensions",
                    value: status?.dimensions ?? "—",
                    note: "One shared mathematical space",
                    icon: Layers3,
                  },
                  {
                    label: "Indexed knowledge",
                    value: stats?.files ?? "—",
                    note: `${stats?.vectors ?? "—"} vectors in unified storage`,
                    icon: Database,
                  },
                  {
                    label: "Inference device",
                    value: status?.device?.toUpperCase() ?? "Not loaded",
                    note: status?.parameter_count
                      ? `${(status.parameter_count / 1e6).toFixed(1)}M loaded parameters`
                      : "Load model to measure parameters",
                    icon: Zap,
                  },
                ].map((s) => (
                  <div className="stat-card" key={s.label}>
                    <div className="row between">
                      <small>{s.label}</small>
                      <s.icon size={17} />
                    </div>
                    <strong>{s.value}</strong>
                    <p>{s.note}</p>
                  </div>
                ))}
              </div>
              <section className="panel">
                <div className="section-title">
                  <div>
                    <span className="eyebrow">THE BIG PICTURE</span>
                    <h2>Different formats. One common language.</h2>
                  </div>
                  <span className="pill">Interactive pipeline</span>
                </div>
                <Diagram />
                <p className="panel-caption">
                  Different types of information can be searched together
                  because the model converts them into a shared mathematical
                  representation.
                </p>
              </section>
              <div className="two-col">
                <section className="panel">
                  <div className="section-title">
                    <h2>Your knowledge, connected</h2>
                    <span className="pill">Live collection</span>
                  </div>
                  {stats && stats.vectors > 0 ? (
                    <ResponsiveContainer width="100%" height={190}>
                      <BarChart
                        data={Object.entries(stats.modalities).map(
                          ([name, count]) => ({
                            name: labels[name as Modality],
                            count,
                          }),
                        )}
                      >
                        <XAxis
                          dataKey="name"
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip />
                        <Bar
                          dataKey="count"
                          fill="#7660df"
                          radius={[6, 6, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <Empty
                      title="A fresh space for your knowledge"
                      text="Upload your first file or load the workplace demo."
                    />
                  )}
                </section>
                <section className="panel try-card">
                  <span className="eyebrow">START WITH A QUESTION</span>
                  <h2>
                    Search for meaning,
                    <br />
                    across every medium.
                  </h2>
                  {examples.map((text) => (
                    <button
                      className="example"
                      key={text}
                      onClick={() => {
                        setQuery(text);
                        setQueryType("text");
                        setPage(2);
                      }}
                    >
                      <Search size={15} />
                      {text}
                      <ArrowRight size={14} />
                    </button>
                  ))}
                </section>
              </div>
              <div className="model-strip">
                <Cpu size={20} />
                <div>
                  <strong>{status?.model || "google/embeddinggemma-2"}</strong>
                  <p>
                    {status?.warning ||
                      "The full multimodal model loads on demand. Weights are cached locally."}
                  </p>
                  {status?.error && (
                    <p className="error-text">{status.error}</p>
                  )}
                </div>
                <button
                  className="secondary"
                  disabled={busy || status?.state === "ready"}
                  onClick={() =>
                    void action(async () => {
                      await post("/api/model/load", {});
                      await refresh();
                    })
                  }
                >
                  {busy ? (
                    <LoaderCircle className="spin" size={16} />
                  ) : (
                    "Load model"
                  )}
                </button>
              </div>
            </>
          )}
          {page === 1 && (
            <>
              <Heading
                eyebrow="BUILD YOUR KNOWLEDGE SPACE"
                title="Upload anything."
                description="Bring your information together. Every file enters the same native embedding pipeline."
              />
              <div
                className="dropzone"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  void upload(e.dataTransfer.files);
                }}
              >
                <div className="upload-icon">
                  <Upload size={30} />
                </div>
                <h2>Drop files into your knowledge space</h2>
                <p>PDF, TXT, JPG, PNG, MP3, WAV, MP4 · Up to 100 MB each</p>
                <label className="primary file-label">
                  Browse files
                  <input
                    type="file"
                    multiple
                    accept=".pdf,.txt,.jpg,.jpeg,.png,.mp3,.wav,.mp4"
                    disabled={busy}
                    onChange={(e) => {
                      if (e.target.files) void upload(e.target.files);
                      e.target.value = "";
                    }}
                  />
                </label>
                <button
                  className="text-button"
                  disabled={busy}
                  onClick={() => void loadDemo()}
                >
                  Or load the workplace demo <ArrowRight size={14} />
                </button>
              </div>
              <section className="panel">
                <div className="section-title">
                  <h2>Live indexing pipeline</h2>
                  <span className="pill">
                    {jobs.length} uploads this session
                  </span>
                </div>
                {!jobs.length && (
                  <p className="muted">
                    Your upload progress will appear here.
                  </p>
                )}
                {jobs.map((j) => (
                  <div className="job" key={j.id}>
                    <div className="row between">
                      <strong>{j.file_name}</strong>
                      <span>
                        {j.stage === "Searchable" ? (
                          <Check size={17} />
                        ) : j.stage !== "Error" ? (
                          <LoaderCircle className="spin" size={17} />
                        ) : null}{" "}
                        {j.stage}
                      </span>
                    </div>
                    <div className="steps">
                      {[
                        "Upload",
                        "Process",
                        "Embed",
                        "Index",
                        "Searchable",
                      ].map((name, i, all) => (
                        <span
                          className={
                            i <= all.indexOf(j.stage) ? "complete" : ""
                          }
                          key={name}
                        >
                          {name}
                          {i < 4 && <ChevronRight size={12} />}
                        </span>
                      ))}
                    </div>
                    {j.error && <p className="error-text">{j.error}</p>}
                    {j.vectors !== undefined && (
                      <small>
                        {j.vectors} embeddings · {j.seconds?.toFixed(2)} seconds
                      </small>
                    )}
                  </div>
                ))}
              </section>
              <div className="section-title">
                <h2>Indexed files</h2>
                <span className="pill">{files.length} originals</span>
              </div>
              <div className="results-grid">
                {files.map((f) => (
                  <ResultCard key={f.url} result={f} />
                ))}
              </div>
              {!files.length && (
                <Empty
                  title="Your library starts here"
                  text="Successfully indexed files appear with original media previews."
                />
              )}
            </>
          )}
          {page === 2 && (
            <>
              <Heading
                eyebrow="CROSS-MODAL DISCOVERY"
                title="Find the idea. In any format."
                description="Ask with text, an image, or audio. Search every type of knowledge at once."
              />
              <section className="search-panel">
                <div className="tabs">
                  {(["text", "image", "audio"] as const).map((m) => {
                    const Icon = icons[m];
                    return (
                      <button
                        className={queryType === m ? "selected" : ""}
                        key={m}
                        onClick={() => {
                          setQueryType(m);
                          setQueryFile(null);
                        }}
                      >
                        <Icon size={16} />
                        {m === "text" ? "Text question" : `${labels[m]} query`}
                      </button>
                    );
                  })}
                </div>
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    void search();
                  }}
                >
                  <div className="search-box">
                    <Search size={23} />
                    {queryType === "text" ? (
                      <input
                        aria-label="Search question"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="What would you like to discover?"
                      />
                    ) : (
                      <input
                        key={queryType}
                        aria-label={`${queryType} query`}
                        type="file"
                        accept={
                          queryType === "image"
                            ? ".jpg,.jpeg,.png"
                            : ".mp3,.wav"
                        }
                        onChange={(e) =>
                          setQueryFile(e.target.files?.[0] || null)
                        }
                      />
                    )}
                    <button className="primary" disabled={busy}>
                      {busy ? (
                        <LoaderCircle className="spin" size={18} />
                      ) : (
                        <ArrowRight size={18} />
                      )}
                      Search
                    </button>
                  </div>
                </form>
                <p className="muted">
                  {queryType === "audio"
                    ? "Use a recording up to 60 seconds. Decoded to mono 16 kHz."
                    : "Native query embeddings · Local retrieval · Real cosine similarities"}
                </p>
                <div className="example-chips">
                  {examples.map((text, i) => (
                    <button
                      key={text}
                      onClick={() => {
                        setQuery(text);
                        setQueryType("text");
                      }}
                    >
                      {
                        [
                          "Worker safety",
                          "Company vehicles",
                          "Employee training",
                        ][i]
                      }
                      <ArrowRight size={12} />
                    </button>
                  ))}
                </div>
              </section>
              <div className="section-title">
                <div className="filters">
                  {(["", "text", "image", "audio", "video"] as const).map(
                    (m) => (
                      <button
                        className={filter === m ? "selected" : ""}
                        key={m}
                        onClick={() => setFilter(m)}
                      >
                        {m ? labels[m] : "All formats"}
                      </button>
                    ),
                  )}
                </div>
                <small>
                  {seconds !== null
                    ? `${results.length} results · ${seconds.toFixed(3)}s`
                    : "Search your unified collection"}
                </small>
              </div>
              <p className="muted small">
                Filter applies to the next search. Scores are alignment, not
                probabilities.
              </p>
              <div className="results-grid">
                {results.map((r) => (
                  <ResultCard key={r.id} result={r} />
                ))}
              </div>
              {!results.length && (
                <Empty
                  title={
                    searched
                      ? "No results in this collection"
                      : "One question. Many perspectives."
                  }
                  text={
                    searched
                      ? "Try another format filter or index files first."
                      : "Actual matches appear with source pages, timestamps, and measured similarities."
                  }
                />
              )}
              {searched && queryType === "text" && (
                <section className="panel">
                  <h2>Keyword search comparison</h2>
                  <p className="muted">
                    Literal word occurrences in text documents. Counts are not
                    semantic similarity scores.
                  </p>
                  {keywords.length ? (
                    keywords.map((r) => (
                      <div className="keyword-row" key={r.id}>
                        <a href={r.url} target="_blank" rel="noreferrer">
                          {r.file_name}
                        </a>
                        <span>{r.keyword_hits} word occurrences</span>
                        <p>{r.text?.slice(0, 220)}</p>
                      </div>
                    ))
                  ) : (
                    <p>No literal word matches.</p>
                  )}
                </section>
              )}
            </>
          )}
          {page === 3 && (
            <>
              <Heading
                eyebrow="SIMPLE AGENTIC RAG"
                title="Answers with evidence."
                description="A small retrieval agent connects your question to cited source material."
              />
              <div className="privacy-note">
                <Sparkles size={18} />
                <span>
                  Retrieval runs locally. Your question and retrieved excerpts
                  go to OpenRouter. Media without readable evidence cannot
                  support a text-only answer.
                </span>
              </div>
              <div className="chat-layout">
                <section className="panel chat-panel">
                  {!messages.length && (
                    <Empty
                      title="What can your knowledge tell you?"
                      text="Ask about your files. The agent can retrieve twice and will identify missing evidence."
                    />
                  )}
                  {messages.map((m, i) => (
                    <div className="chat-turn" key={i}>
                      <div className="user-message">{m.question}</div>
                      <div className="answer-message">
                        <span className="eyebrow">
                          <Sparkles size={13} /> EVIDENCE-BASED ANSWER
                        </span>
                        <p>{m.answer}</p>
                        <div className="source-links">
                          {m.sources.map((s, n) => (
                            <a
                              href={
                                s.url +
                                (s.page
                                  ? `#page=${s.page}`
                                  : s.start_time !== undefined
                                    ? `#t=${s.start_time}`
                                    : "")
                              }
                              key={s.id}
                              target="_blank"
                              rel="noreferrer"
                            >
                              [{n + 1}] {s.file_name}
                            </a>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                  <form
                    className="chat-input"
                    onSubmit={(e) => {
                      e.preventDefault();
                      void action(async () => {
                        const r = await post<{
                          answer: string;
                          sources: Result[];
                          stages: Record<string, unknown>[];
                        }>("/api/chat", { question });
                        setMessages((m) => [...m, { question, ...r }]);
                        setQuestion("");
                      });
                    }}
                  >
                    <input
                      aria-label="Chat question"
                      required
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="Ask about your knowledge…"
                    />
                    <button
                      aria-label="Send question"
                      className="primary"
                      disabled={busy}
                    >
                      {busy ? (
                        <LoaderCircle size={17} className="spin" />
                      ) : (
                        <ArrowRight size={18} />
                      )}
                    </button>
                  </form>
                </section>
                <aside className="panel trace-panel">
                  <span className="eyebrow">AGENT ACTIVITY</span>
                  <h2>Follow the evidence</h2>
                  {!messages.length ? (
                    <p className="muted">
                      Actual stages appear after your question completes.
                    </p>
                  ) : (
                    messages.at(-1)?.stages.map((s, i) => (
                      <div className="trace" key={i}>
                        <span>{i + 1}</span>
                        <div>
                          <strong>{String(s.stage)}</strong>
                          {Object.entries(s)
                            .filter(([key]) => key !== "stage")
                            .map(([key, value]) => (
                              <p key={key}>
                                {key.replaceAll("_", " ")}:{" "}
                                {typeof value === "number"
                                  ? value.toFixed(3)
                                  : String(value)}
                              </p>
                            ))}
                        </div>
                      </div>
                    ))
                  )}
                  {messages.at(-1)?.sources.map((s, i) => (
                    <details key={s.id}>
                      <summary>
                        [{i + 1}] {s.file_name} · rank {s.rank}
                      </summary>
                      <p>
                        {s.text ||
                          "No transcript or caption. This media match cannot establish an answer."}
                      </p>
                    </details>
                  ))}
                </aside>
              </div>
            </>
          )}
          {page === 4 && (
            <>
              <Heading
                eyebrow="A FIVE-MINUTE EXPLORATION"
                title="From information to understanding."
                description="Follow the science behind a shared embedding space, one step at a time."
              />
              <div className="learning-tabs">
                {["Inputs", "Encode", "Store", "Retrieve", "Answer"].map(
                  (name, i) => (
                    <button
                      className={step === i ? "selected" : ""}
                      onClick={() => setStep(i)}
                      key={name}
                    >
                      <span>0{i + 1}</span>
                      {name}
                    </button>
                  ),
                )}
              </div>
              <section className="panel learning-panel">
                <span className="eyebrow">STEP 0{step + 1}</span>
                <h2>{learning[step][0]}</h2>
                <p>{learning[step][1]}</p>
                {step === 2 ? (
                  <div className="projection">
                    <svg
                      viewBox="0 0 680 240"
                      role="img"
                      aria-label="Conceptual modality vectors"
                    >
                      {Array.from({ length: 40 }, (_, i) => (
                        <circle
                          key={i}
                          cx={130 + (i % 10) * 43 + Math.sin(i * 3) * 60}
                          cy={
                            70 + Math.floor(i / 10) * 33 + Math.cos(i * 2) * 27
                          }
                          r={5}
                          fill={
                            ["#7660df", "#d29435", "#2e9c84", "#ca6688"][i % 4]
                          }
                        />
                      ))}
                    </svg>
                    <p>
                      Conceptual approximate 2D projection — illustrative
                      points, not measured outputs or the original
                      768-dimensional space.
                    </p>
                  </div>
                ) : (
                  <Diagram />
                )}
                <div className="row between">
                  <a
                    className="text-button"
                    href="/guide.html"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <BookOpen size={15} />
                    Printable five-page guide
                  </a>
                  <button
                    className="secondary"
                    onClick={() => setStep((step + 1) % 5)}
                  >
                    Next step
                    <ArrowRight size={15} />
                  </button>
                </div>
              </section>
              <section className="panel">
                <div className="section-title">
                  <div>
                    <span className="eyebrow">TRY THE SCIENCE</span>
                    <h2>Different words. Related meaning?</h2>
                  </div>
                  <FlaskConical size={24} />
                </div>
                <p className="muted">
                  Edit the sentences and measure genuine cosine similarities
                  with the symmetric SentenceSimilarity task.
                </p>
                <div className="experiment-inputs">
                  {texts.map((t, i) => (
                    <label key={i}>
                      TEXT {String.fromCharCode(65 + i)}
                      <textarea
                        value={t}
                        onChange={(e) =>
                          setTexts((current) =>
                            current.map((v, j) =>
                              i === j ? e.target.value : v,
                            ),
                          )
                        }
                      />
                    </label>
                  ))}
                </div>
                <button
                  className="primary"
                  disabled={busy}
                  onClick={() =>
                    void action(async () => {
                      const r = await post<{
                        similarities: number[][];
                        seconds: number;
                      }>("/api/experiment", texts);
                      setMatrix(r.similarities);
                      setSeconds(r.seconds);
                    })
                  }
                >
                  {busy ? (
                    <LoaderCircle size={16} className="spin" />
                  ) : (
                    <FlaskConical size={16} />
                  )}
                  Measure similarity
                </button>
                {matrix && (
                  <>
                    <table>
                      <caption>
                        Measured cosine similarities · {seconds?.toFixed(3)}s
                      </caption>
                      <thead>
                        <tr>
                          <th>Text</th>
                          {matrix.map((_, i) => (
                            <th key={i}>{String.fromCharCode(65 + i)}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {matrix.map((row, i) => (
                          <tr key={i}>
                            <th>{String.fromCharCode(65 + i)}</th>
                            {row.map((n, j) => (
                              <td key={j}>{n.toFixed(4)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="muted">
                      Scores are not probabilities. No ranking outcome is
                      assumed.
                    </p>
                  </>
                )}
              </section>
              <section className="panel">
                <h2>Explore across modalities</h2>
                <div className="demo-links">
                  {[
                    {
                      title: "Image → text",
                      text: "Use a workplace image as a query to find policy passages.",
                      kind: "image",
                    },
                    {
                      title: "Audio → text",
                      text: "Use a spoken recording to find related text.",
                      kind: "audio",
                    },
                    {
                      title: "Text → video",
                      text: "Find timestamped clips about safety procedures.",
                      kind: "text",
                    },
                  ].map((d) => (
                    <button
                      key={d.title}
                      onClick={() => {
                        setQueryType(d.kind as "text" | "image" | "audio");
                        setFilter(d.kind === "text" ? "video" : "text");
                        setQuery("Workers demonstrating safety procedures");
                        setPage(2);
                      }}
                    >
                      <strong>{d.title}</strong>
                      <p>{d.text}</p>
                      <ArrowRight size={18} />
                    </button>
                  ))}
                </div>
              </section>
            </>
          )}
          <footer className="page-footer">
            <span>ONE SPACE / EmbeddingGemma 2 laboratory</span>
            <span>Local embeddings. Connected knowledge.</span>
          </footer>
        </div>
      </main>
      {busy && (
        <div className="working" role="status">
          <LoaderCircle size={16} className="spin" />
          Processing · first inference may download weights
        </div>
      )}
    </div>
  );
}
