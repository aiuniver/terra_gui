<template>
  <div class="panel-settings">
    {{ type }}
    <SettingInput v-if="type === 'input'" :selected="getSelected" />
    <SettingHandler v-if="type === 'handler'" :selected="getSelected" />
    <SettingMiddle v-if="type === 'middle'" :selected="getSelected" />
    <SettingMiddle v-if="type === 'middle'" :selected="getSelected" />
    <SettingOutput v-if="type === 'output'" :selected="getSelected" />
    <SettingEmpty v-if="!type" />
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import SettingEmpty from './SettingEmpty';
import SettingInput from './SettingInput';
import SettingHandler from './SettingHandler';
import SettingMiddle from './SettingMiddle';
import SettingOutput from './SettingOutput';

export default {
  name: 'DatasetSettings',
  components: { SettingEmpty, SettingInput, SettingMiddle, SettingHandler, SettingOutput },
  data: () => ({
    show: true,
    ops: {
      scrollPanel: {
        scrollingX: true,
        scrollingY: false,
      },
    },
  }),
  computed: {
    ...mapGetters({
      getSelected: 'create/getSelected',
    }),
    type() {
      return this?.getSelected?.type || '';
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

<style lang="scss">
.panel-settings {
}
</style>