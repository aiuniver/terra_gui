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
      formsHandler: 'datasets/getFormsHandler',
    }),
  },
};
</script>