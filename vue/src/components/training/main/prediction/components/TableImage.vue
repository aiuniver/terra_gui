<template>
  <img class="table-prediction__image" :src="require('@/../public/imgs/' + path)" alt="img" />
</template>

<script>
export default {
  name: 'TableImage',
  props: {
    image: {
      type: [Object, Array],
      default: () => {},
    },
  },
  computed: {
    path() {
      return this.image?.data || '';
    },
    type() {
      return this.image?.type || null;
    },
  },
};
</script>

<style lang="scss" scoped>
.table-prediction {
  &__image {
    width: 270px;
    height: 160px;
  }
}
</style>
