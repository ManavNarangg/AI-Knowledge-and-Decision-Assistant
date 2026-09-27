/* Knowledge assistant embed script.
 * Usage: <script src="https://YOUR-HOST/embed.js" data-assistant-token="ast_..." async></script>
 * Adds a floating chat button that opens the organization's assistant in a sandboxed iframe.
 * Optional attributes: data-position="left", data-color="#2f5bea", data-title="Ask us".
 */
(function () {
  var script = document.currentScript || document.querySelector("script[data-assistant-token]");
  if (!script) return;
  var token = script.getAttribute("data-assistant-token");
  if (!token) { console.warn("[assistant] data-assistant-token is missing"); return; }
  var origin = new URL(script.src).origin;
  var color = script.getAttribute("data-color") || "#2f5bea";
  var side = script.getAttribute("data-position") === "left" ? "left" : "right";
  var title = script.getAttribute("data-title") || "Ask a question";

  var button = document.createElement("button");
  button.type = "button";
  button.setAttribute("aria-label", title);
  button.title = title;
  button.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';
  button.style.cssText = "position:fixed;bottom:20px;" + side + ":20px;width:58px;height:58px;border-radius:50%;border:0;" +
    "background:" + color + ";box-shadow:0 6px 20px rgba(0,0,0,.25);cursor:pointer;z-index:2147483646;display:grid;place-items:center";

  var frame = document.createElement("iframe");
  frame.src = origin + "/a/" + encodeURIComponent(token);
  frame.title = title;
  frame.setAttribute("sandbox", "allow-scripts allow-same-origin allow-forms allow-popups");
  frame.setAttribute("referrerpolicy", "no-referrer");
  frame.style.cssText = "position:fixed;bottom:90px;" + side + ":20px;width:min(400px,calc(100vw - 40px));" +
    "height:min(620px,calc(100vh - 120px));border:0;border-radius:14px;box-shadow:0 12px 40px rgba(0,0,0,.25);" +
    "z-index:2147483647;display:none;background:#fff";

  button.addEventListener("click", function () {
    frame.style.display = frame.style.display === "none" ? "block" : "none";
  });

  function mount() { document.body.appendChild(frame); document.body.appendChild(button); }
  if (document.body) mount(); else document.addEventListener("DOMContentLoaded", mount);
})();
