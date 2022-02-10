<template>
  <div>
    <template v-for="(data, index) of formsHandler">
      <TAutoFieldHandler v-bind="data" :parameters="{}" :key="index" :idKey="'key_' + index" root @change="change" />
    </template>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import TAutoFieldHandler from '@/components/new/blocks/TAutoFieldHandler';

export default {
  components: {
    TAutoFieldHandler
  },
  computed: {
    ...mapGetters({
      formsHandler: 'datasets/getFormsHandler',
    }),
  },
};
</script>