import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Renders LLM answer text as markdown (GFM: tables, lists, code, links).
// Styling lives in the `.md-body` rules in index.css. Raw HTML is not enabled,
// so model output can't inject markup.
export function Markdown({ children }: { children: string }) {
  return (
    <div className="md-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}
