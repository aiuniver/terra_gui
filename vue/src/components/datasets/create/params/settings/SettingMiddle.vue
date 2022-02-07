<template>
  <div>
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
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
export default {
  components: {},
  computed: {
    ...mapGetters({
      getDefault: 'create/getDefault',
    }),
    input() {
      return this.getDefault('input');
    },
  },
};
</script>