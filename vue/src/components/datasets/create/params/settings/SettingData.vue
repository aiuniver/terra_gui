<template>
  <div class="panel-input">
    <div class="panel-input__view">
      <Cards>
        <template v-for="(file, i) of getFile">
          <CardFile v-if="file.type === 'folder'" v-bind="file" :key="'files_' + i" />
          <CardTable v-if="file.type === 'table'" v-bind="file" :key="'files_' + i" />
        </template>
      </Cards>
    </div>
    <div class="panel-input__forms">
      <t-field label="Данные">
        <d-select v-model="sel" :list="listFiles"></d-select>
      </t-field>
      {{ selected }}
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import Cards from '../card/Cards';
import CardFile from '../card/CardFile';
import CardTable from '../card/CardTable';
export default {
  components: {
    CardFile,
    CardTable,
    Cards,
  },
  props: {
    selected: {
      type: Object,
      default: () => ({}),
    },
  },
  data: () => ({
    sel: {},
  }),
  computed: {
    ...mapGetters({
      getFiles: 'createDataset/getFiles',
      getFileManager: 'createDataset/getFileManager',
    }),
    listFiles() {
      return this.getFileManager.map(i => ({ label: i.title, value: i.value }));
    },
    getFile() {
      return this.getFiles.map(i => ({ label: i.title, value: i.value }));
    },
  },
};
</script>

<style lang="scss">
.panel-input {
  &__view {
    position: relative;
    margin-bottom: 20px;
    height: 170px;
  }
  &__forms {
  }
}
</style>