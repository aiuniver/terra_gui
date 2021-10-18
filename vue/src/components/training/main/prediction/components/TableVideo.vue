<template>
  <div id="app">
    <Media :kind="'video'" :controls="true" :src="src" :style="{ width: '200px' }" :key="src"></Media>
  </div>
</template>

<script>
import Media from '@dongido/vue-viaudio';

export default {
  components: {
    Media,
  },
  props: {
    value: { type: String, default: '' },
    update: {
      type: String,
      default: '',
    },
  },
  computed: {
    src() {
       return `/_media/blank/?path=${this.value}&r=${this.update}`;
    },
  },
  data: () => ({
    autoStart: false,
  }),
};
</script>