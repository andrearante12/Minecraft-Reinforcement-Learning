/* Live dashboard: same renderer as the overview site, WebSocket source. */
(function () {
  "use strict";
  if (window.NetViz) {
    window.NetViz.mount({ source: "ws" });
  }
})();
