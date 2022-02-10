<template>
  <div>
    <t-field>
      <d-input-text />
    </t-field>
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