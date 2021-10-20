<template>
  <div class="t-text-segmented" :style="{width: block_width}">
    <div class="t-text-segmented__content">
      <div v-for="({ tags, word }, index) of arrText" :key="'word_' + index" class="t-text-segmented__word">
        <at-tooltip v-if="!tags.includes('p1')">
          <div class="t-text-segmented__text">{{ word }}</div>
          <template slot="content">
            <div class="t-text-segmented__colors" v-for="color of tags" :key="`colors_${color}`" :style="`background-color: rgb(${rgb(color)});`">{{ color }}</div>
          </template>
        </at-tooltip>
        <div v-else class="t-text-segmented__text">{{ word }}</div>
        <div
          v-for="(tag, i) of tags"
          :key="`tags_${index}_${i}`"
          class="t-text-segmented__line"
          :style="`background-color: rgb(${rgb(tag)});`"
        ></div>
      </div>
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
    block_width: {
      type: String,
      default: '400px'
    }
  },
  computed: {
    tags() {
      return this.tags_color?.[this.layer] || {};
    },
    arrText() {
      let text = this.value.replaceAll(' ', '');
      const arr = text.match(/(<[sp][0-9]>)+([^<\/>]+)(<\/[sp][0-9]>)+/g); // eslint-disable-line
      return arr.map(item => {
        return this.convert(item);
      });
    },
  },
  methods: {
    rgb(tag) {
      const arr = this.tags[tag] || [];
      return arr.join(' ');
    },
    convert(str) {
      str = str.replace(/(<\/[^>]+>)+/g, '');
      const word = str.replace(/(<[^>]+>)+/g, '');
      str = str.replace(/></g, ',');
      const tags = str.match(/<(.+)>/)[1].split(',');
      return { tags, word };
    },
  },
};
</script>

<style lang="scss" scoped>
.t-text-segmented {
  width: 400px;
  padding: 10px;
  position: relative;

  &__content {
    width: 100%;
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-start;
    position: relative;
    align-items: flex-start;
    text-align: left;
    user-select: none;
    // gap: 5px;
  }
  &__word {
    // margin: 0 5px 0 0;
    display: flex;
    flex: 1 1;
    flex-direction: column;
  }
  &__text {
    padding: 0 5px 0 0;
  }
  &__colors {
    height: 20px;
    width: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    margin: 3px 0;
    font-size: 1.2em;
    font-weight: 700;
  }
  &__line {
    display: block;
    height: 2px;
    // margin: 0 0 1px 0;
    // width: calc(100% + 10px);
  }
}
</style>