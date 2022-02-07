<template>
  <div class="block-file">
    <div :class="['block-file__header', { toggle: !toggle }]" @click="(toggle = !toggle), $emit('toggle', toggle)">
      <i
        class="block-file__header--icon"
      ></i>
      {{ text }}
    </div>
    <div v-show="toggle" class="block-file__body">
      <scrollbar>
      <files-menu v-model="filesSource" />
      </scrollbar>
    </div>
  </div>
</template>

<script>
export default {
  name: "BlockFiles",
  data: () => ({
    toggle: true,
    nodes: [
      {
        title: "Cars",
        type: "folder",
        isExpanded: false,
        children: [
          { title: "BMW.jpg", type: "image" },
          { title: "AUDI.jpg", type: "image" },
        ],
      },
      {
        title: "Music",
        type: "folder",
        isExpanded: true,
        children: [
          { title: "1.mp3", type: "audio" },
          { title: "song.wav", type: "audio" },
        ],
      },
      {
        title: "Text",
        type: "folder",
        isExpanded: false,
        children: [{ title: "Table", type: "text" }],
      },
    ],
  }),
  computed: {
    text() {
      return this.toggle ? "Выбор папки/файла" : "";
    },
    filesSource: {
      set(value) {
        this.$store.dispatch('datasets/setFilesSource', value)
      },
      get() {
        return this.$store.getters['datasets/getFilesSource']
      }

    },
  },
};
</script>

<style lang="scss" scoped>
.block-file {
  width: 100%;
  height: 100%;
  position: relative;
  &__header {
    position: absolute;
    height: 24px;
    width: 100%;
    top: 0;
    background: #242f3d;
    font-family: Open Sans;
    font-style: normal;
    font-weight: normal;
    font-size: 12px;
    line-height: 16px;
    display: flex;
    align-items: center;
    text-align: center;
    color: #ffffff;
    padding: 4px 16px;
    overflow: hidden;
    text-overflow: ellipsis;
    &.toggle {
      padding: 0px;
      flex: 0 0 24px;
    }
    &--icon {
      cursor: pointer;
      display: inline-block;
      position: absolute;
      width: 7px;
      height: 11px;
      right: 9px;
      background-repeat: no-repeat;
      background-image: url("data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iOCIgaGVpZ2h0PSIxMiIgdmlld0JveD0iMCAwIDggMTIiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik02LjcwOTk4IDAuNzA5OTk2QzYuMzE5OTggMC4zMTk5OTYgNS42ODk5OCAwLjMxOTk5NiA1LjI5OTk4IDAuNzA5OTk2TDAuNzA5OTggNS4zQzAuMzE5OTggNS42OSAwLjMxOTk4IDYuMzIgMC43MDk5OCA2LjcxTDUuMjk5OTggMTEuM0M1LjY4OTk4IDExLjY5IDYuMzE5OTggMTEuNjkgNi43MDk5OCAxMS4zQzcuMDk5OTggMTAuOTEgNy4wOTk5OCAxMC4yOCA2LjcwOTk4IDkuODlMMi44Mjk5OCA2TDYuNzA5OTggMi4xMkM3LjA5OTk4IDEuNzMgNy4wODk5OCAxLjA5IDYuNzA5OTggMC43MDk5OTZaIiBmaWxsPSIjQTdCRUQzIi8+Cjwvc3ZnPgo=");
      transition: transform 0.7s;
      .toggle & {
        // transform: rotate(180deg);
        transform: rotateY(180deg);
        right: 7px;
      }
    }
  }
  &__body {
    padding-top: 30px;
    height: 100%;
  }
}
</style>