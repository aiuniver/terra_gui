<template>
  <div class="panel-input">
    <div class="panel-input__view">
      <template v-for="(file, i) of mixinFiles">
        <CardFile v-if="file.type === 'folder'" v-bind="file" :key="'files_' + i" />
        <CardTable v-if="file.type === 'table'" v-bind="file" :key="'files_' + i" />
      </template>
    </div>
    <div class="panel-input__forms">
      <t-field label="Данные">
        <d-select :value="1" :list="listFiles"></d-select>
      </t-field>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import CardFile from '@/components/datasets/paramsFull/components/card/CardFile.vue';
import CardTable from '@/components/datasets/paramsFull/components/card/CardTable';
export default {
  components: {
    CardFile,
    CardTable,
  },
  props: {
    selected: {
      type: Object
    }
  },
  computed: {
    ...mapGetters({
      getFileManager: 'createDataset/getFileManager',
    }),
    mixinFiles() {
      return this.getFileManager.map(e => {
        console.log(e)
        return {
          id: e.id,
          cover: e.cover,
          label: e.title,
          type: e.type,
          table: e.table,
          value: e.path,
        };
      });
    },
    listFiles() {
      return this.getFileManager.map(i => ({label: i.title, value: i.value}))
    }
  },
};
</script>

<style lang="scss">
.panel-input {
  &__view {
    display: flex;
    margin-bottom: 20px;
  }
  &__forms {

  }
}
</style>