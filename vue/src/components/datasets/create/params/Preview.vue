<template>
  <div class="preview">
    <div class="preview__files">
      <files-menu v-model="filesSource" />
    </div>
    <div class="preview__title">Предпросмотр</div>
    <div class="preview__cards">
      <template v-for="(file, i) of mixinFiles">
        <CardFile v-if="file.type === 'folder'" v-bind="file" :key="'files_' + i" />
        <CardTable v-if="file.type === 'table'" v-bind="file" :key="'files_' + i"  />
      </template>
    </div>
    <div class="preview__title">Параметры</div>
  </div>
</template>

<script>
import CardFile from '@/components/datasets/paramsFull/components/card/CardFile.vue';
import CardTable from '@/components/datasets/paramsFull/components/card/CardTable';
import { mapGetters } from 'vuex';
export default {
  components: {
    CardFile,
    CardTable,
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
    }),
    mixinFiles() {
      return this.getFileManager.map(e => {
        return {
          id: e.id,
          cover: e.cover,
          label: e.label,
          type: e.type,
          table: e.table,
          value: e.path,
        };
      });
    },
    filesSource: {
      set(value) {
        this.$store.dispatch('datasets/setFilesSource', value);
      },
      get() {
        return this.getFileManager;
      },
    },
  },
};
</script>

<style lang="scss" scoped>
@import '@/assets/scss/variables/default.scss';
.preview {
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
    display: flex;
    padding: 30px 0;
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