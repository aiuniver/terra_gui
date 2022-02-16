<template>
  <div>
    <template v-for="(data, index) of formsHandler">
      <t-auto-field-handler v-bind="data" :parameters="{}" :key="index" :idKey="'key_' + index" root @change="change" />
    </template>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
export default {
  components: {},
  computed: {
    ...mapGetters({
      getSelected: 'create/getSelected',
      getFileManager: 'createDataset/getFileManager',
      getDefault: 'create/getDefault',
    }),
  },
};
</script>