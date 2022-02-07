<template>
  <div class="panel-settings">
    {{ type }}
    <template v-if="type === 'input'">
      <template v-for="(file, i) of mixinFiles">
        <CardFile v-if="file.type === 'folder'" v-bind="file" :key="'files_' + i" />
        <CardTable v-if="file.type === 'table'" v-bind="file" :key="'files_' + i" />
      </template>
    </template>
    <template v-if="type === 'handler'">
      <template v-for="(data, index) of formsHandler">
        <t-auto-field-handler v-bind="data" :parameters="{}" :key="index" :idKey="'key_' + index" root @change="change" />
      </template>
    </template>
    <template v-if="type === 'middle'">
      <template v-for="(data, index) of input">
        <t-auto-field
          v-bind="data"
          :parameters="parameters"
          :key="'inputData.color' + index"
          :idKey="'key_' + index"
          :id="index + 3"
          :update="mixinUpdateDate"
          :isAudio="isAudio"
          root
          @multiselect="mixinUpdate"
          @change="mixinChange"
        />
      </template>
    </template>
    <template v-if="type === 'output'">
      <template v-for="(data, index) of formsHandler">
        <t-auto-field-handler v-bind="data" :parameters="{}" :key="index" :idKey="'key_' + index" root @change="change" />
      </template>
    </template>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import CardFile from '@/components/datasets/paramsFull/components/card/CardFile.vue';
import CardTable from '@/components/datasets/paramsFull/components/card/CardTable';
export default {
  name: 'DatasetSettings',
  components: { CardFile, CardTable },
  data: () => ({
    show: true,
    ops: {
      scrollPanel: {
        scrollingX: true,
        scrollingY: false,
      },
    },
    colors: [
      '#1ea61d',
      '#a51da6',
      '#0d6dea',
      '#fecd05',
      '#d72239',
      '#054f1d',
      '#630e76',
      '#031e70',
      '#b78b01',
      '#660634',
      '#86e372',
      '#e473d0',
      '#6bb5f9',
      '#ffe669',
      '#f38079',
    ],
    table: {},
  }),
  computed: {
    ...mapGetters({
      getSelected: 'create/getSelected',
      getFileManager: 'createDataset/getFileManager',
      getDefault: 'create/getDefault',
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
    formsHandler() {
      return this.$store.getters['datasets/getFormsHandler'];
    },
    type() {
      return this?.getSelected?.type || '';
    },

    //___ input
    input() {
      return this.getDefault('input');
    },
  },
  created() {
    const files = this.$store.getters['datasets/getFilesSource'];
    console.log(files);
    this.table = files
      .filter(item => item.type === 'table')
      .reduce((obj, item) => {
        obj[item.title] = [];
        return obj;
      }, {});
  },
  methods: {
    change({ id, value, name }) {
      const index = this.handlers.findIndex(item => item.id === id);
      if (name === 'name') {
        this.handlers[index].name = value;
      }
      if (name === 'type') {
        this.handlers[index].type = value;
      }
      if (this.handlers[index]) {
        this.handlers[index].parameters[name] = value;
      }
      this.handlers = [...this.handlers];
    },
    select(id) {
      this.handlers = this.handlers.map(item => {
        item.active = item.id === id;
        return item;
      });
    },
    deselect() {
      this.handlers = this.handlers.map(item => {
        item.active = false;
        return item;
      });
    },
    handleAdd() {
      if (!this.show) return;
      console.log(this.table);
      this.deselect();
      let maxID = Math.max(0, ...this.handlers.map(o => o.id));
      this.handlers.push({
        id: maxID + 1,
        name: 'Name_' + (maxID + 1),
        active: true,
        color: this.colors[this.handlers.length],
        layer: (this.handlers.length + 1).toString(),
        type: '',
        table: JSON.parse(JSON.stringify(this.table)),
        parameters: {},
      });
      console.log(this.handlers);
    },
    handleClick(event, id) {
      if (event === 'remove') {
        this.deselect();
        this.handlers = this.handlers.filter(item => item.id !== id);
      }
      console.log(event);
      if (event === 'copy') {
        this.deselect();
        const copy = JSON.parse(JSON.stringify(this.handlers.filter(item => item.id == id)));
        let maxID = Math.max(0, ...this.handlers.map(o => o.id));
        copy[0].id = maxID + 1;
        (copy[0].name = 'Name_' + (maxID + 1)), (this.handlers = [...this.handlers, ...copy]);
      }
    },
  },
};
</script>
