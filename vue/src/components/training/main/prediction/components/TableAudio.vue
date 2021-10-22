<template>
  <div class="item">
    <div class="audio">
      <div class="audio__card">
        <div :class="['audio__btn', { pause: playing }]" @click="handleClick"></div>
        <av-bars v-if="!initial"
          :canv-width="95"
          :canv-height="25"
          :bar-width="2"
          bar-color="#2B5278"
          canv-fill-color="#242F3D"
          ref-link="audio"
        ></av-bars>
        <div v-else></div>
        <p class="audio__time">{{ formatTime(curTime) }} / {{ duration }}</p>
        <audio ref="audio" :src="src"
        preload="none"
        @canplay="canplay"
        @timeupdate="curTime = $event.target.currentTime"
        @play="playing = true"
        @pause="playing = false"
        ></audio>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TableAudio',
  props: {
    value: {
      type: String,
      default: '',
    },
    update: {
      type: String,
      default: '',
    },
  },
  data: () => ({
    loaded: false,
    curTime: 0,
    playing: false,
    initial: true
  }),
  computed: {
    src() {
      return `/_media/blank/?path=${this.value}&r=${this.update}`;
    },
    duration() {
      if (!this.loaded) return '00:00'
      return this.formatTime(this.$refs.audio.duration)
    }
  },
  methods: {
    handleClick() {
      if (this.initial) this.initial = false
      setTimeout(() => {
        this.playing ? this.$refs.audio.pause() : this.$refs.audio.play();
      }, 1)
    },
    canplay() {
      this.loaded = true
    },
    formatTime(time) {
      const minutes = Math.floor(time.toFixed() / 60);
      const seconds = time.toFixed() % 60
      return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
    }
  }
};
</script>

<style lang="scss" scoped>
.item {
  padding: 10px;
}
.audio {
  padding: 10px;
  background: #242f3d;
  border: 1px solid #6c7883;
  border-radius: 4px;
  position: relative;
  overflow: hidden;
  min-width: 180px;
  width: min-content;
  &__btn {
    cursor: pointer;
    grid-row: 1 / 3;
    width: 52px;
    height: 52px;
    background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNTIiIGhlaWdodD0iNTIiIHZpZXdCb3g9IjAgMCA1MiA1MiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTI2LjAwMDIgNC4zMzM1QzE0LjA0MDIgNC4zMzM1IDQuMzMzNSAxNC4wNDAyIDQuMzMzNSAyNi4wMDAyQzQuMzMzNSAzNy45NjAyIDE0LjA0MDIgNDcuNjY2OCAyNi4wMDAyIDQ3LjY2NjhDMzcuOTYwMiA0Ny42NjY4IDQ3LjY2NjggMzcuOTYwMiA0Ny42NjY4IDI2LjAwMDJDNDcuNjY2OCAxNC4wNDAyIDM3Ljk2MDIgNC4zMzM1IDI2LjAwMDIgNC4zMzM1Wk0yMS42NjY4IDMzLjU4MzVWMTguNDE2OEMyMS42NjY4IDE3LjUyODUgMjIuNjg1MiAxNy4wMDg1IDIzLjQwMDIgMTcuNTUwMkwzMy41MTg1IDI1LjEzMzVDMzQuMTAzNSAyNS41NjY4IDM0LjEwMzUgMjYuNDMzNSAzMy41MTg1IDI2Ljg2NjhMMjMuNDAwMiAzNC40NTAyQzIyLjY4NTIgMzQuOTkxOCAyMS42NjY4IDM0LjQ3MTggMjEuNjY2OCAzMy41ODM1WiIgZmlsbD0iIzY1QjlGNCIvPgo8L3N2Zz4K');
    &.pause {
      background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNTIiIGhlaWdodD0iNTIiIHZpZXdCb3g9IjAgMCA1MiA1MiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIxLjY2NjYgMzQuNjY2N0MyMi44NTgzIDM0LjY2NjcgMjMuODMzMyAzMy42OTE3IDIzLjgzMzMgMzIuNVYxOS41QzIzLjgzMzMgMTguMzA4MyAyMi44NTgzIDE3LjMzMzMgMjEuNjY2NiAxNy4zMzMzQzIwLjQ3NDkgMTcuMzMzMyAxOS40OTk5IDE4LjMwODMgMTkuNDk5OSAxOS41VjMyLjVDMTkuNDk5OSAzMy42OTE3IDIwLjQ3NDkgMzQuNjY2NyAyMS42NjY2IDM0LjY2NjdaTTI1Ljk5OTkgNC4zMzMzNEMxNC4wMzk5IDQuMzMzMzQgNC4zMzMyNSAxNC4wNCA0LjMzMzI1IDI2QzQuMzMzMjUgMzcuOTYgMTQuMDM5OSA0Ny42NjY3IDI1Ljk5OTkgNDcuNjY2N0MzNy45NTk5IDQ3LjY2NjcgNDcuNjY2NiAzNy45NiA0Ny42NjY2IDI2QzQ3LjY2NjYgMTQuMDQgMzcuOTU5OSA0LjMzMzM0IDI1Ljk5OTkgNC4zMzMzNFpNMjUuOTk5OSA0My4zMzMzQzE2LjQ0NDkgNDMuMzMzMyA4LjY2NjU5IDM1LjU1NSA4LjY2NjU5IDI2QzguNjY2NTkgMTYuNDQ1IDE2LjQ0NDkgOC42NjY2OCAyNS45OTk5IDguNjY2NjhDMzUuNTU0OSA4LjY2NjY4IDQzLjMzMzMgMTYuNDQ1IDQzLjMzMzMgMjZDNDMuMzMzMyAzNS41NTUgMzUuNTU0OSA0My4zMzMzIDI1Ljk5OTkgNDMuMzMzM1pNMzAuMzMzMyAzNC42NjY3QzMxLjUyNDkgMzQuNjY2NyAzMi40OTk5IDMzLjY5MTcgMzIuNDk5OSAzMi41VjE5LjVDMzIuNDk5OSAxOC4zMDgzIDMxLjUyNDkgMTcuMzMzMyAzMC4zMzMzIDE3LjMzMzNDMjkuMTQxNiAxNy4zMzMzIDI4LjE2NjYgMTguMzA4MyAyOC4xNjY2IDE5LjVWMzIuNUMyOC4xNjY2IDMzLjY5MTcgMjkuMTQxNiAzNC42NjY3IDMwLjMzMzMgMzQuNjY2N1oiIGZpbGw9IiM2NUI5RjQiLz4KPC9zdmc+Cg==');
    }
  }
  &__card {
    display: grid;
    grid-template: 1fr 19px / 52px 1fr;
    column-gap: 10px;
    align-items: center;
  }
  &__time {
    text-align: left;
    font-size: 9px;
    color: #A7BED3;
  }
}

</style>