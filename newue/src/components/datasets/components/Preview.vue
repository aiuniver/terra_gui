<template>
  <div class="preview">
    <scrollbar :ops="ops">
      <div class="preview-list flex align-center">
        <div class="preview-list__item mr-4" v-for="item in list" :key="JSON.stringify(item)">
          <img :src="item.path" :alt="item.label" />
          <p>{{ item.label }}</p>
        </div>
      </div>
    </scrollbar>
  </div>
</template>

<script>
export default {
  name: 'BasePreview',
  props: {
    list: {
      type: Array,
      default: () => [],
    },
  },
  data: () => ({
    ops: {
      scrollPanel: {
        scrollingX: true,
        scrollingY: false,
      },
    },
  }),
};
</script>

<style lang="scss" scoped>
.preview {
  &-list {
    &__item {
      width: 140px;
      border-radius: 4px;
      border: 1px solid $color-blue;
      overflow: hidden;
      p {
        padding: 0 10px;
        color: $color-gray-blue;
      }
      img {
        width: 100%;
        height: 100px;
        object-fit: cover;
      }
    }
  }
}
</style>