<template>
  <div class="t-text-segmented" title="">
    <div class="t-text-segmented__content" v-html="text">
      <!-- <p class="t-text-segmented__text" v-html="text">{{ }}</p> -->
    </div>
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
    tags() {
      return this.tags_color?.[this.layer] || {};
    },
    text() {
      let text = this.value.replaceAll(/></g, '+');
      text = text.replace(/<\/[^>]+>/g, '<>');
      text = text.replace(/<[^>]+>/g, this.convert);
      text = text.replaceAll('<>', '</div></div>');
      return text;
    },
  },
  methods: {
    rgb(tag) {
      console.log(tag);
      const arr = this.tags[tag] || [];
      return arr.join(' ');
    },
    convert(str) {
      str = str.replace(/[<>]/g, '');
      str = str.split('+');
      const tags = str
      str = str
        .map(item => {
          return `<div class="t-text-segmented__line ${item}" style="background-color: rgb(${this.rgb(item)});"></div>`;
        })
        .join('');

      return `<div class="t-text-segmented__word" title="${tags.join(' ')}">${str}<div class="t-text-segmented__text">&nbsp;`;
    },
  },
};
</script>

<style lang="scss">
.t-text-segmented {
  width: 400px;
  padding: 10px;
  position: relative;

  &__content {
    width: 100%;
    display: flex;
    flex-wrap: wrap;
    // justify-content: flex-start;
    position: relative;
    align-items: flex-start;
    text-align: start;
  }
  &__word {
    margin: 0 5px 0 0;
    display: flex;
    flex-direction: column-reverse;
  }
  &__text {
  }
  &__line {
    display: block;
    height: 2px;
    margin: 0 0 1px 0;
    width: calc(100% + 5px);
  }
}
</style>