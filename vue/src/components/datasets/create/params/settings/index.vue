<template>
  <div class="panel-settings">
    <SettingData v-if="type === 'data'" :selected="selected" />
    <SettingHandler v-if="type === 'handler'" :selected="selected" :forms="forms" />
    <SettingOutput v-if="['output', 'input'].includes(type)" :selected="selected" />
    <SettingEmpty v-if="!type" />
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import SettingEmpty from './SettingEmpty';
import SettingData from './SettingData';
import SettingHandler from './SettingHandler';
import SettingOutput from './SettingOutput';

export default {
  name: 'DatasetSettings',
  components: { SettingEmpty, SettingData, SettingHandler, SettingOutput },
  props: {
    state: {
      type: Number,
      default: 1,
    },
  },
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
      getHandler: 'createDataset/getHandler',
    }),
    forms() {
      const type = this.state === 2 ? 'inputs' : 'outputs';
      return this.getHandler?.[type] || [];
    },
    selected() {
      const len = this.getSelected.filter(i => i.selected).length;
      return len === 1 ? this.getSelected.find(i => i.selected) : {};
    },
    type() {
      return this?.selected?.type || '';
    },
  },
  methods: {},
};
</script>

<style lang="scss">
.panel-settings {
}
</style>