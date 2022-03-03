<template>
  <div class="t-cards" :style="style" @wheel.prevent="wheel">
    <scrollbar :ops="ops" ref="scrollCards">
      <div class="t-cards__items">
        <div class="t-cards__items--item">
          <slot> </slot>
        </div>
      </div>
    </scrollbar>
  </div>
</template>

<script>
export default {
  data: () => ({
    wight: 0,
    ops: {
      scrollPanel: {
        scrollingX: true,
        scrollingY: false,
      },
    },
  }),
  computed: {
    style() {
      return { wight: this.wight + "px" };
    },
  },
  mounted() {
    this.wight = this.$el.clientWidth;
    console.log(this.$el.clientWidth);
  },
  methods: {
    wheel(e) {
      e.stopPropagation();
      this.$refs.scrollCards.scrollBy(
        {
          dx: e.wheelDelta,
        },
        200
      );
    },
  },
};
</script>

<style lang="scss" scoped>
.t-cards {
  width: 100%;
  height: 100%;
  position: absolute;

  &__items {
    display: flex;
    // position: absolute;
    // width: 100%;
    height: 100%;
    // top: 10px;
    &--item {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: row;
      align-items: center;
      padding-top: 5px;
    }
  }
}
</style>