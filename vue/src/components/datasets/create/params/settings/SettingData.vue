<template>
  <div class="panel-input">
    <div v-if="isShow" class="panel-input__overlay">Нельзя изменить</div>
    <div class="panel-input__view mb-10">
      <t-field label="Предпросмотр"></t-field>
      <Cards>
        <template v-for="(file, i) of getFile">
          <CardFile v-if="file.type === 'folder'" v-bind="file" :key="'files_' + i" />
          <CardTable v-if="file.type === 'table'" v-bind="file" :key="'files_' + i" />
        </template>
        <div v-if="!getFile.length" class="panel-input__empty">Нет данных</div>
      </Cards>
    </div>
    <div class="panel-input__forms">
      <t-field label="Название">
        <d-input-text :value="name" placeholder="Название блока" @change="onChange"></d-input-text>
      </t-field>
      <t-field label="Входные данные">
        <d-multi-select :value="getValueData" placeholder="Данные" :list="listFiles" @change="onFile" @clear="onClearFile" />
      </t-field>
      <t-field v-if="isTable" label="Таблица">
        <d-multi-select :value="getCollumData" placeholder="Колонки" :list="getCollums" @change="onCollum" @clear="onClearCollum" />
      </t-field>
      {{ selected }}
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
    state: {
      type: Number,
      default: 1,
    },
  },
  data: () => ({}),
  computed: {
    ...mapGetters({
      getFiles: 'createDataset/getFiles',
      getFileManager: 'createDataset/getFileManager',
    }),
    name() {
      return this.selected?.name || '';
    },
    isShow() {
      const { type, created } = this.selected
      return type === 'data' && created === 'input' && this.state === 3
    },
    listFiles() {
      let arr = this.getFileManager
        .map(({ value, title, type, path, data }) => ({ label: title, value, type, path, data }))
        .filter(i => i.type === this.type || !this.type);
      if (this.type === 'table') arr = arr.filter(i => this.file === i.label);
      return arr;
    },
    getFile() {
      return this.type === 'table' ? this.getFiles.filter(i => this.file === i.label) : this.getFiles.filter(i => this.data.includes(i.label));
    },
    getValueData() {
      return this.type === 'table' ? [this.file] : this.selected?.parameters?.data || [];
    },
    getCollumData() {
      return this.selected?.parameters?.data || [];
    },
    collums() {
      return this.getFile?.[0]?.table || [];
    },
    getCollums() {
      return this.collums.map(i => ({ label: i[0], value: i[0] }));
    },
    getParametrs() {
      return this?.selected?.parameters || {};
    },
    id() {
      return this.selected.id;
    },
    type() {
      return this.selected?.parameters?.type || '';
    },
    file() {
      return this.selected?.parameters?.file || '';
    },
    data() {
      return this.selected?.parameters?.data || [];
    },
    isTable() {
      return Boolean(this.type === 'table');
    },
  },
  methods: {
    ...mapActions({
      setParameters: 'create/setParameters',
      editBlock: 'create/editBlock',
    }),
    onChange({ value }) {
      const block = { ...this.selected };
      block.name = value;
      this.editBlock(block);
    },
    onFile({ label, type }) {
      let data = this.data;
      let file = '';
      if (type === 'table') {
        data = [];
        if (this.file) {
          file = '';
          type = '';
        } else {
          file = label;
        }
      } else {
        if (data.includes(label)) {
          data = data.filter(i => i !== label);
        } else {
          data.push(label);
        }
        if (!data.length) type = '';
      }

      const parameters = { ...this.parameters, data, type, file };
      this.setParameters({ id: this.id, parameters });
    },
    onCollum({ label }) {
      let data = this.data;
      let type = this.type;
      let file = this.file;
      if (data.includes(label)) {
        data = data.filter(i => i !== label);
      } else {
        data.push(label);
      }
      const parameters = { data, file, type };
      this.setParameters({ id: this.id, parameters });
    },
    onClearFile() {
      const parameters = { ...this.parameters, data: [], type: '', file: '' };
      this.setParameters({ id: this.id, parameters });
    },
    onClearCollum() {
      let type = this.type;
      let file = this.file;
      const parameters = { ...this.parameters, data: [], type, file };
      this.setParameters({ id: this.id, parameters });
    },
  },
};
</script>

<style lang="scss">
.panel-input {
  position: relative;
  &__view {
    position: relative;
    margin-bottom: 20px;
    height: 170px;
  }
  &__forms {
  }
  &__overlay {
    z-index: 2;
    background-color: #0e162170;
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
  }
}
</style>