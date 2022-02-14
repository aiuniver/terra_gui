<template>
  <div>
    <t-field>
      <d-input-text :value="value" @change="onChange"></d-input-text>
    </t-field>
    {{ selected }}
  </div>
</template>

<script>
import { mapGetters, mapActions } from 'vuex';
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
    value() {
      return this.selected?.name || '';
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
  },
};
</script>