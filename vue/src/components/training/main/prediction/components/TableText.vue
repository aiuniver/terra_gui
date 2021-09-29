<template>
  <p> {{ value }}</p>
</template>

<script>
export default {
  name: 'TableText',
  props: {
    value: {
      type: String,
      default: '',
    },
    color_mark: {
      type: Array,
      default: () => [],
    },
  },

  computed: {
    // text() {
    //   const temp = this.value
    //     .split(' ')
    //     .map(el => {
    //       const tag = el.substring(0, el.indexOf('>') + 1);
    //       let temp = `<span style="color: rgb(${this.color_mark[tag].join(',')})">${el.replace(/<[^>]+>/g, '')}</span>`;
    //       return temp;
    //     })
    //     .join(' ');
    //   return temp;
    // },
    // type() {
    //   return this.data?.type || null;
    // },
  },
};
</script>
