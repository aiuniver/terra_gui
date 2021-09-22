<template>
  <div class="item">
    <div class="item__wrapper">
      <img :src="'data:image/png;base64,' + image" alt="" />
      <div class="item__wrapper__info">
        <div
          v-for="({ label, value }, i) of info"
          :key="'info_' + i"
          class="param"
        >
          <label>{{ label }}: </label>
          <span v-text="value" />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "CardImage",
  props: {
    image: {
      type: String,
      default: ''},
    info: {
      type: Array,
      default: () => [],
    },
  },
};
</script>

<style lang="scss" scoped>
.item {
  width: 200px;
  padding: 0 5px 10px 5px;
  &__wrapper {
    width: 100%;
    height: 100%;
    padding: 5px;
    background: #242f3d;
    border-radius: 4px;
    box-shadow: 0 2px 10px 0 rgb(0 0 0 / 25%);
    overflow: hidden;
    user-select: none;
    img {
      width: 100% !important;
      height: auto !important;
      vertical-align: top;
    }
    &__info {
      line-height: 1;
      margin: 5px 0 0 0;
      .param {
        label {
          color: #a7bed3;
          font-size: 0.75rem;
          line-height: 1.25;
        }
        span {
          font-size: 0.875rem;
          font-weight: 600;
          line-height: 1.25;
        }
      }
    }
  }
}
</style>