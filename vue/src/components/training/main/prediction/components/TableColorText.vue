<template>
  <p>
    {{ text }}
  </p>
</template>

<script>
export default {
  name: 'TableText',
  props: {
    data: {
      type: [Object, Array],
      default: () => {},
    },
  },

  computed: {
    tagsColor() {
      return this.data?.tags_color || '';
    },
    text() {
      const pattern = new RegExp('<\\W>\\w</\\W>', 'g');
      const temp = this.data?.data.replace(pattern, function (str, p1) {
        return `<span style="color: ${this.tagsColor[p1]}">${str}</span>`;
      });

      return temp || '';
    },
    type() {
      return this.data?.type || null;
    },
  },
};
</script>

<style lang="scss" scoped>
.table-prediction {
}
</style>
