<template>
  <svg width="100%" height="100%">
    <!-- <circle :cx="x" :cy="y" :r="2 * scale" fill="none" :style="line.style" /> -->
    <path :d="line.data" :style="line.style"/>
  </svg>
</template>

<script>
export default {
  props: {
    x: Number,
    y: Number,
    scale: Number,
  },
  computed: {
    line() {
      const dist = 100 * this.scale
      return {
        data: `M ${this.x},${this.y + dist} ${this.x},${this.y - dist} M ${this.x + dist},${this.y} ${this.x - dist},${this.y}`,
        style: {
          stroke: '#6c7883',
          strokeWidth: 0.5 * this.scale,
        },
      };
    },
  },
};
</script>

<style scoped>
</style>
