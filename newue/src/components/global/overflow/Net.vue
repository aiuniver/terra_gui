<template>
  <svg width="100%" height="100%">
    <!-- <circle :cx="1" :cy="endpointY" :r="1" :fill="color" /> -->
    <line :x1="endpointX" :y1="0" :x2="endpointX" :y2="lenghtPoint" :style="line.style"></line>
    <line :x1="endpointX" :y1="lengthHeight" :x2="endpointX" :y2="lengthHeight - lenghtPoint" :style="line.style"></line>

    <line :x1="0" :y1="endpointY" :x2="lenghtPoint" :y2="endpointY" :style="line.style"></line>

    <line :x1="lengthWidth" :y1="endpointY" :x2="lengthWidth - lenghtPoint" :y2="endpointY" :style="line.style"></line>

    <line :x1="x" :y1="y - line.dist" :x2="x" :y2="y + line.dist" :style="line.style"></line>
    <line :x1="x - line.dist" :y1="y" :x2="x + line.dist" :y2="y" :style="line.style"></line>

  </svg>
</template>

<script>
import { debounce } from '@/utils/blocks/utils';
export default {
  props: {
    x: Number,
    y: Number,
    scale: Number,
  },
  data: () => ({
    lenghtPoint: 10,
    show: 0.6,
    cell: 20,
    width: 0,
    height: 0,
    color: '#6c7883',
    debounce: null,
  }),
  computed: {
    endpointX() {
      return this.x > 0 ? (this.x > this.width ? this.width - 1 : this.x) : 0;
    },
    endpointY() {
      return this.y > 0 ? (this.y > this.height ? this.height - 1 : this.y) : 0;
    },
    pixel() {
      return this.cell * this.scale;
    },
    lengthHeight() {
      return this.height;
    },
    lengthWidth() {
      return this.width;
    },
    style() {
      return { stroke: '#6c78832f', strokeWidth: 0.5 * this.scale };
    },
    line() {
      const dist = 60 * this.scale;
      return {
        // data: `M ${this.x},${this.y + dist} ${this.x},${this.y - dist} M ${this.x + dist},${this.y} ${this.x - dist},${
        //   this.y
        // }`,
        dist,
        style: {
          stroke: this.color,
          strokeWidth: 0.5 * this.scale,
        },
      };
    },
  },
  methods: {
    event(e) {
      this.debounce(e);
    },
  },
  mounted() {
    this.width = this.$el?.clientWidth;
    this.height = this.$el?.clientHeight;
    console.log(this.width, this.height);
    this.debounce = debounce(() => {
      this.width = this.$el?.clientWidth;
      this.height = this.$el?.clientHeight;
    }, 200);
    window.addEventListener('resize', this.event);
  },
  destroyed() {
    window.removeEventListener('resize', this.event);
  },
};
</script>

<style scoped>
</style>
