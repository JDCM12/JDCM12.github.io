(function () {
  "use strict";

  const root = document.getElementById("demo-root");
  const inline = document.getElementById("demo-data");

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function duration(seconds) {
    if (!Number.isFinite(Number(seconds))) return "";
    const total = Math.round(Number(seconds));
    const minutes = Math.floor(total / 60);
    const remainder = String(total % 60).padStart(2, "0");
    return `${minutes}:${remainder}`;
  }

  function render(data) {
    const items = [...(data.items || [])].sort((left, right) => {
      const languageOrder = { English: 0, Chinese: 1 };
      const sampleNumber = (item) => {
        const match = String(item.id || "").match(/(\d+)$/);
        return match ? Number(match[1]) : 999;
      };
      return (
        (languageOrder[left.language] ?? 99) -
          (languageOrder[right.language] ?? 99) ||
        sampleNumber(left) - sampleNumber(right)
      );
    });
    if (!items.length) {
      root.innerHTML = '<p class="error">No local sample is configured.</p>';
      return;
    }
    let selected = 0;

    function renderSelected() {
      const item = items[selected];
      root.innerHTML = `
        <nav class="sample-nav" aria-label="Song sample selection">
          ${items.map((sample, index) => `
            <button class="sample-tab ${index === selected ? "active" : ""}"
              type="button" data-index="${index}"
              aria-selected="${index === selected}">
              <span>${String(index + 1).padStart(2, "0")}</span>
              <strong>${escapeHtml(sample.title)}</strong>
              <small>${escapeHtml(sample.language || "")}</small>
            </button>
          `).join("")}
        </nav>
        <div class="sample-shell">
          <section class="prompt-card">
            <span class="label">${escapeHtml(item.language || "Sample")} · ${String(selected + 1).padStart(2, "0")} / ${items.length}</span>
            <h2>${escapeHtml(item.title)}</h2>
            <p class="prompt">${escapeHtml(item.prompt)}</p>
            <details open>
              <summary>Lyrics</summary>
              <pre class="lyrics">${escapeHtml(item.lyrics)}</pre>
            </details>
          </section>
          <section class="song-list" aria-label="System comparison">
            ${item.methods.map((method) => `
              <article class="song-card">
                <div class="song-meta">
                  <h3>${escapeHtml(method.label)}</h3>
                  <span class="method-tag">${escapeHtml(method.description || "paper system")}</span>
                </div>
                <audio controls preload="none" src="${escapeHtml(method.file)}">
                  Your browser does not support audio playback.
                </audio>
                <p class="audio-caption">Same prompt and lyrics · generated audio (max 4:00)</p>
              </article>
            `).join("")}
          </section>
        </div>
      `;

      root.querySelectorAll(".sample-tab").forEach((button) => {
        button.addEventListener("click", () => {
          selected = Number(button.dataset.index);
          renderSelected();
        });
      });
      root.querySelectorAll("audio").forEach((audio) => {
        audio.addEventListener("play", () => {
          root.querySelectorAll("audio").forEach((other) => {
            if (other !== audio) other.pause();
          });
        });
      });
    }

    renderSelected();
  }

  let inlineData = null;
  try {
    inlineData = JSON.parse(inline.textContent);
  } catch (error) {
    root.innerHTML = '<p class="error">The embedded demo manifest is invalid.</p>';
  }

  const data = window.PAPER_DEMOS || inlineData;
  if (data) {
    render(data);
  }
})();
