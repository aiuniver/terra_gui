<template>
  <p v-html="text"></p>
</template>

<script>
export default {
  name: 'TableTag',
  props: {
    data: {
      type: [Object, Array],
      default: () => {},
    },
  },

  computed: {
    tagsColor() {
      return this.data?.tags_color || [];
    },
    text() {
      const temp = this.data.data
        .split(' ')
        .map(el => {
          const tag = el.substring(0, el.indexOf('>') + 1);
          let temp = `<span style="color: rgb(${this.tagsColor[tag].join(',')})">${el.replace(/<[^>]+>/g, '')}</span>`;
          return temp;
        })
        .join(' ');
      return temp;
    },
    type() {
      return this.data?.type || null;
    },
  },
};
</script>
