<template>
  <div class="panel-settings">
    {{ type }}
    <SettingData v-if="type === 'data'" :selected="getSelected" />
    <SettingHandler v-if="type === 'handler'" :selected="getSelected" />
    <SettingOutput v-if="['output', 'input'].includes(type)" :selected="getSelected" />
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
  methods: {
  },
};
</script>

<style lang="scss">
.panel-settings {
}
</style>