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
        <d-multi-select :value="getValueData" placeholder="Данные" :list="listFiles" @change="onFile" />
      </t-field>
      {{ getParametrs }}
    </div>
  </div>
</template>

<script>
import { mapGetters, mapActions } from 'vuex';
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
      return this.getFiles.filter(i => this.items.includes(i.label));
    },
    getValueData() {
      return this.selected?.parameters?.items || [];
    },
    getParametrs() {
      return this?.selected?.parameters || {};
    },
    id() {
      return this.selected.id;
    },
    items() {
      return this.selected?.parameters?.items || [];
    },
  },
  methods: {
    ...mapActions({
      setParameters: 'create/setParameters',
    }),
    onFile(data) {
      let items = this.items;
      if (items.includes(data.label)) {
        items = items.filter(i => i !== data.label);
      } else {
        items.push(data.label);
      }
      const parameters = { ...this.parameters, items };
      this.setParameters({ id: this.id, parameters });
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