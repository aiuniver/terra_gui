<template>
  <div :class="['t-predict-image', { 't-predict-image--large': isLarge }]" v-outside="outside">
    <!-- <div v-if="isLarge && show" class="t-predict-image__mask" @click="click(false)"></div>
    <div v-if="isLarge && show" class="t-predict-image__fixed">
      <i class="ci-icon ci-close_big" @click="click(false)"/>
      <img width="auto" :height="600" :src="src" :alt="'value'" :key="src" />
    </div> -->
    <img width="auto" :height="isLarge ? 300 : 120" :src="src" :alt="'value'" :key="src" @click="click(true)" />
  </div>
</template>
<script>
export default {
  name: 't-table-image',
  props: {
    value: {
      type: String,
      default: '',
    },
    update: String,
    size: String,
  },
  data: () => ({
    show: false,
  }),
  computed: {
    src() {
      return `/_media/blank/?path=${this.value}&r=${this.update}`;
    },
    isLarge() {
      return this.size === 'large';
    },
  },
  methods: {
    click(value) {
      if (this.isLarge) {
        this.show = value;
      }
      if (value) this.$store.dispatch('trainings/setLargeImg', this.src)
    },
    outside() {
      this.show = false;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-predict-image {
  min-width: 200px;
  display: flex;
  justify-content: center;
  align-items: center;
  // padding: 0 20px;
  &--large {
    height: 300px;
    min-width: 500px;
    cursor: pointer;
  }
  &__fixed {
    // display: none;
    z-index: 701;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    padding: 20px;
    background-color: #17212b;
    border-radius: 6px;
    border: 1px solid #6c7883;
    cursor: default;
    img {
      border-radius: 6px;
    }
    i {
      position: absolute;
      // height: 30px;
      // width: 30px;
      top: -20px;
      right: -20px;
      font-size: 30px;
      background-color: #17212b;
      border-radius: 50%;
      border: 1px solid #6c7883;
      padding: 5px;
      color: #6c7883;
      cursor: pointer;
      &:hover {
        color: #b5bbc0;
      }
    }
  }
  &__mask {
    position: fixed;
    z-index: 700;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    background-color: #0e1621e6;
    cursor: default;
  }
  // &__image {}
}
</style>
