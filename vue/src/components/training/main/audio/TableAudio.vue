<template>
<div class="audio">
  <av-waveform
    :canv-width="120"
    :canv-height="23"
    played-line-color="#2B5278"
    noplayed-line-color="#2B5278"
    :playtime="false"
    :playtime-slider="false"
    :canv-top="true"
    audio-class="audio__player"
    canv-class="audio__player"
    canv-fill-color="#2B5278"
    :audio-src="require('@/../public/sounds/sound1.mp3')"
  ></av-waveform>
</div>
</template>

<script>
export default {
  name: "TableAudio"
}
</script>

<style lang="scss">
.audio{
  background: #242F3D;
  border: 1px solid #6C7883;
  audio::-webkit-media-controls-panel{
    background: #242F3D;
    border-radius: 0;
  }
  audio::-webkit-media-controls-play-button{
    //background: #65B9F4;
  }
  audio::-webkit-media-controls-timeline-container,
  audio::-webkit-media-controls-timeline,
  audio::-webkit-media-controls-volume-slider-container,
  audio::-webkit-media-controls-volume-control-container,
  audio::-webkit-media-controls-volume-slider,
  audio::-webkit-media-controls-seek-back-button,
  audio::-webkit-media-controls-seek-forward-button,
  audio::-webkit-media-controls-fullscreen-button,
  audio::-webkit-media-controls-rewind-button,
  audio::-webkit-media-controls-return-to-realtime-button,
  audio::-webkit-media-slider-thumb,
  audio::-webkit-media-controls-toggle-closed-captions-button{
    display: none !important;
  }
  audio::-webkit-media-controls-enclosure {
    border-radius: 0;
    background-color: #242F3D;
  }
  audio::-webkit-media-controls-play-button, video::-webkit-media-controls-play-button {
    -webkit-appearance: media-play-button;
    display: flex;
    flex: none;
    border: none;
    box-sizing: border-box;
    border-radius: 20px;
    width: 43px;
    height: 43px;
    line-height: 30px;
    margin-left: 9px;
    margin-right: 9px;
    padding: 0;
    background-color: #65B9F4;
    color: inherit;
}
  audio::-webkit-media-controls-current-time-display, video::-webkit-media-controls-current-time-display,
  audio::-webkit-media-controls-time-remaining-display, video::-webkit-media-controls-time-remaining-display {
    -webkit-appearance: media-current-time-display;
    -webkit-user-select: none;
    flex: none;
    display: flex;
    border: none;
    cursor: default;
    height: 30px;
    margin: 43px 0 0 0;
    padding: 0;
    padding-left: 10px;
    line-height: 19px;
    font-family: Open Sans;
    font-size: 9px;
    font-weight: normal;
    font-style: normal;
    color: #A7BED3;
    letter-spacing: normal;
    word-spacing: normal;
    text-transform: none;
    text-indent: 0;
    text-shadow: none;
    text-decoration: none;
  }
}
</style>