<template>
  <div class="item">
    <div class="audio">
      <div class="audio__card">
        <av-waveform
          :canv-width="500"
          :canv-height="23"
          played-line-color="#65B9F4"
          noplayed-line-color="#2B5278"
          :played-line-width="0"
          :playtime="false"
          :canv-top="true"
          canv-class="custom-player"
          canv-fill-color="#2B5278"
          :audio-src="src"
          :key="update"
        ></av-waveform>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'AudioCard',
  props: {
    value: {
      type: String,
      default: '',
    },
  },
  computed: {
    src() {
      return `/_media/blank/?path=${this.value}&r=${Date.now()}`;
    },
  },
  mounted() {
    this.$el.querySelector('audio').setAttribute('controlsList', 'nodownload');
  },
};
</script>

<style lang="scss">
.audio {
  padding: 10px;
  background: #242f3d;
  border: 1px solid #6c7883;
  border-radius: 4px;
  width: 600px;
  position: relative;
  overflow: hidden;
  &__card {
    .custom-player {
      left: 60px;
      top: 50%;
      transform: translateY(-50%);
      position: absolute;
      z-index: 2;
    }
    audio {
      width: 200px;
      margin-left: -10px;
      padding: 0;
      display: flex;
      flex-wrap: nowrap;
      justify-content: flex-start;
    }
    audio::-webkit-media-controls-panel {
      border-radius: 4px;
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
    audio::-webkit-media-controls-toggle-closed-captions-button {
      display: none !important;
    }
    audio::-webkit-media-controls-enclosure {
      border-radius: 0;
      display: flex;
      flex-wrap: nowrap;
      justify-content: flex-start;
      padding: 0;
      width: 200px;
      background-color: transparent;
    }
    audio::-webkit-media-controls-play-button,
    video::-webkit-media-controls-play-button {
      -webkit-appearance: media-play-button;
      display: flex;
      flex: none;
      border: none;
      box-sizing: border-box;
      border-radius: 20px;
      width: 43px;
      height: 43px;
      line-height: 30px;
      margin-right: 9px;
      padding: 0;
      background-color: #65b9f4;
      color: inherit;
    }
    audio::-webkit-media-controls-current-time-display,
    video::-webkit-media-controls-current-time-display,
    audio::-webkit-media-controls-time-remaining-display,
    video::-webkit-media-controls-time-remaining-display {
      -webkit-appearance: media-current-time-display;
      -webkit-user-select: none;
      flex: none;
      display: flex;
      border: none;
      cursor: default;
      height: 30px;
      margin: 43px 0 0 0;
      padding: 0;
      line-height: 19px;
      font-family: Open Sans;
      font-size: 9px;
      font-weight: normal;
      font-style: normal;
      color: #a7bed3;
      letter-spacing: normal;
      word-spacing: normal;
      text-transform: none;
      text-indent: 0;
      text-shadow: none;
      text-decoration: none;
    }
  }
}
</style>