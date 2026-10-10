// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import App from "./App";

vi.mock("@xyflow/react", () => ({
  ReactFlow: ({
    nodes,
  }: {
    nodes: { id: string; data: { label: string } }[];
  }) => (
    <div>
      {nodes.map((n) => (
        <span key={n.id}>{n.data.label}</span>
      ))}
    </div>
  ),
  Background: () => null,
  Controls: () => null,
}));
const match = {
  id: "policy:0",
  file_name: "safety.pdf",
  modality: "text",
  rank: 1,
  similarity: 0.8123,
  text: "Wear a hard hat in the loading area.",
  url: "/media/policy.pdf",
  page: 2,
};
let requests: { path: string; init?: RequestInit }[];
beforeEach(() => {
  requests = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string, init?: RequestInit) => {
      requests.push({ path, init });
      let body: unknown = {};
      if (path === "/api/model/status")
        body = {
          model: "google/embeddinggemma-2",
          state: "not_loaded",
          device: null,
          dimensions: 768,
          parameter_count: null,
        };
      if (path === "/api/stats")
        body = {
          files: 0,
          vectors: 0,
          modalities: { text: 0, image: 0, audio: 0, video: 0 },
        };
      if (path === "/api/files") body = [];
      if (path === "/api/upload") body = { job_id: "job-1" };
      if (path === "/api/jobs/job-1")
        body = {
          id: "job-1",
          file_name: "policy.txt",
          stage: "Searchable",
          vectors: 1,
          seconds: 0.15,
        };
      if (path === "/api/search/text")
        body = { results: [match], keyword_results: [match], seconds: 0.235 };
      if (path === "/api/chat")
        body = {
          answer: "Wear a hard hat [1].",
          sources: [match],
          stages: [
            { stage: "Retrieve", query: "safety", results: 1 },
            { stage: "Cited answer", external_provider: "OpenRouter" },
          ],
        };
      if (path === "/api/experiment")
        body = {
          similarities: [
            [1, 0.7, 0.1],
            [0.7, 1, 0.2],
            [0.1, 0.2, 1],
          ],
          seconds: 0.3,
        };
      return { ok: true, json: async () => body };
    }),
  );
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("five connected application pages", () => {
  it("renders real backend status, navigation, learning steps, and theme", async () => {
    render(<App />);
    expect(await screen.findByText("Not loaded")).toBeTruthy();
    expect(screen.getByRole("heading", { name: /One AI model/ })).toBeTruthy();
    for (const [nav, heading] of [
      ["Upload anything", "Upload anything."],
      ["Semantic search", "Find the idea. In any format."],
      ["Evidence chat", "Answers with evidence."],
      ["How it works", "From information to understanding."],
    ]) {
      fireEvent.click(screen.getByRole("button", { name: nav }));
      expect(screen.getByRole("heading", { name: heading })).toBeTruthy();
    }
    fireEvent.click(screen.getByRole("button", { name: /03.*Store/ }));
    expect(
      screen.getByText(/Conceptual approximate 2D projection/),
    ).toBeTruthy();
    expect(
      screen
        .getByRole("link", { name: /Printable five-page/ })
        .getAttribute("href"),
    ).toBe("/guide.html");
    fireEvent.click(screen.getByRole("button", { name: "Dark appearance" }));
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("uploads original files and displays actual job progress", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Upload anything" }));
    const file = new File(["Helmet policy"], "policy.txt", {
      type: "text/plain",
    });
    fireEvent.change(screen.getByLabelText("Browse files"), {
      target: { files: [file] },
    });
    expect(await screen.findByText("policy.txt")).toBeTruthy();
    await waitFor(
      () =>
        expect(screen.getByText("1 embeddings · 0.15 seconds")).toBeTruthy(),
      { timeout: 3000 },
    );
    expect(
      requests.find((r) => r.path === "/api/upload")?.init?.body,
    ).toBeInstanceOf(FormData);
  });

  it("sends text and modality to search, then renders measured scores and pages", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Semantic search" }));
    fireEvent.change(screen.getByLabelText("Search question"), {
      target: { value: "Workers wearing helmets" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Text" }));
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText("0.812")).toBeTruthy();
    expect(screen.getByText("Page 2")).toBeTruthy();
    expect(screen.getByText("1 results · 0.235s")).toBeTruthy();
    const request = requests.find((r) => r.path === "/api/search/text");
    expect(JSON.parse(String(request?.init?.body))).toEqual({
      text: "Workers wearing helmets",
      modality: "text",
    });
  });

  it("renders cited chat answers with sources and actual stages", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Evidence chat" }));
    fireEvent.change(screen.getByLabelText("Chat question"), {
      target: { value: "What safety equipment?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    expect(await screen.findByText("Wear a hard hat [1].")).toBeTruthy();
    expect(screen.getByText("Cited answer")).toBeTruthy();
    expect(screen.getByText("external provider: OpenRouter")).toBeTruthy();
    expect(screen.getByRole("link", { name: "[1] safety.pdf" })).toBeTruthy();
  });
});
