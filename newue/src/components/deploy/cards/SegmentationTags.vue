<template>
  <div class="card">
    <div class="card__text">
      <div class="card__text--line" v-for="(color, tag, index) in tags" :key="'tag-line_' + index">
        <span :style="{ 'background-color': rgb(color) }">{{ tag }}</span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'SegmentationTags',
  props: {
    tags: {
      type: Object,
      default: () => ({}),
    },
  },
  methods: {
    rgb(arr) {
      let rgbString = '#';
      for (let i in arr) {
        let hex = arr[i].toString(16);
        rgbString += hex.length == 1 ? '0' + hex : hex;
      }

      return rgbString;
    },
  },
  data: () => ({
    ops: {
      scrollPanel: {
        scrollingX: false,
        scrollingY: true,
      },
    },
  }),
};
</script>

<style lang="scss" scoped>
.card {
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
  min-height: 54px;
  padding: 6px;
  font-size: 14px;
  line-height: 21px;
  color: #65b9f4;
  &__text {
    display: flex;
    padding: 0 10px;
    &--line {
      margin-top: 2px;
      span {
        width: 30px;
        text-align: center;
        display: inline-block;
        padding: 0 3px;
        border-radius: 4px;
        margin: 0 10px 0 0;
      }
    }
  }
}
</style>