(function () {
  "use strict";
  document.querySelectorAll(".clip-slot").forEach(function (slot) {
    var video = slot.querySelector("video");
    if (!video) {
      slot.classList.add("is-empty");
      return;
    }
    slot.classList.add("is-empty");
    function show() {
      if (video.videoWidth > 0) slot.classList.remove("is-empty");
    }
    video.addEventListener("error", function () { slot.classList.add("is-empty"); });
    video.addEventListener("loadeddata", show);
    if (video.readyState >= 2) show();
  });
})();
