<template>
  <div class="state-two">
    <div class="state-two__cards">
      <Cards>
        <template v-for="(file, i) of getFiles">
          <CardFile v-if="file.type === 'folder'" v-bind="file" :key="'files_' + i" />
          <CardTable v-if="file.type === 'table'" v-bind="file" :key="'files_' + i" />
        </template>
      </Cards>
    </div>
    <div class="state-two__title mb-2">Файлы</div>
    <div class="state-two__files">
      <files-menu v-model="filesSource" />
    </div>
  </div>
</template>

<script>
import Cards from './card/Cards';
import CardFile from './card/CardFile';
import CardTable from './card/CardTable';
import { mapGetters } from 'vuex';
export default {
  components: {
    CardFile,
    CardTable,
    Cards
  },
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
  computed: {
    ...mapGetters({
      getFileManager: 'createDataset/getFileManager',
      getFiles: 'createDataset/getFiles',
    }),
    filesSource: {
      set(value) {
        this.$store.dispatch('datasets/setFilesSource', value);
      },
      get() {
        return this.getFileManager;
      },
    },
  },
  methods: {
    
  },
};
</script>

<style lang="scss" scoped>
@import '@/assets/scss/variables/default.scss';
.state-two {
  &__title {
    display: flex;
    align-items: center;
    height: 48px;
    border-top: 1px solid black;
    border-bottom: 1px solid black;
    font-style: normal;
    font-weight: 600;
    font-size: 14px;
    line-height: 140%;
  }
  &__cards {
    position: relative;
    height: 170px;
  }
  &__files {
    height: 300px;
    overflow: auto;
  }
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