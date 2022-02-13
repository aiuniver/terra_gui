<template>
  <div>
    <t-field>
      <v-select></v-select>
    </t-field>
    {{parameters}}
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
export default {
  components: {},
  props: {
    selected: {
      type: Object,
      default: () => ({}),
    },
  },
  computed: {
    ...mapGetters({
      getFileManager: 'createDataset/getFileManager',
      getDefault: 'create/getDefault',
    }),
    parameters() {
      return this?.selected?.parameters || {};
    },
  },
};
</script>