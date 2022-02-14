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
      <t-field label="Входные данные">
        <d-auto-complete :value="getValueData" placeholder="Архитектуры" :list="listFiles" @change="onArchitectures" />
      </t-field>
      {{ getParametrs }}
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
      editBlock: 'create/editBlock',
    }),
    listFiles() {
      return this.getFileManager.map(i => ({ label: i.title, value: i.value }));
    },
    getFile() {
      return this.getFiles.map(i => ({ label: i.title, value: i.value }));
    },
    getValueData () {
      const value = this.selected?.parametrs?.filename || ''
      return this.listFiles.find(i => i.value === value)
    },
    getParametrs() {
      return this?.selected?.parametrs || {};
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