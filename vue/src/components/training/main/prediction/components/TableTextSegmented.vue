<template>
  <div class="t-text-segmented">
    <p class="t-text-segmented__text" v-html="text">{{}}</p>
  </div>
</template>

<script>
export default {
  name: 'TableTextSegmented',
  props: {
    value: {
      type: String,
      default: '',
    },
    color_mark: {
      type: Array,
      default: () => [],
    },
    tags_color: {
      type: Object,
      default: () => {},
    },
    layer: {
      type: String,
      default: '',
    },
  },

  computed: {
    text() {
      let text = this.value.replace(/<\/[^>]+>/g, '</span>');
      const layer = this.tags_color[this.layer];
      for (let key in layer) {
        const [r, g, b] = layer[key];
        text = text.replaceAll(key, `<span style="background-color: rgb(${r} ${g} ${b}); border-radius: 4px; padding: 0 5px 0 5px;">`);
      }
      return text;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-text-segmented {
  width: 300px;
  padding: 10px;
  &__text {
    text-align: left;
  }
}
</style>