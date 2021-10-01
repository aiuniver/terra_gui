<template>
  <div id="app">
    <Media :kind="'video'" :controls="true" :src="src"       :style="{width: '200px'}"></Media>
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
  },
  computed: {
    src() {
      return `/_media/blank/?path=${this.value}`;
    },
  },
  data: () => ({
    autoStart: false,
  }),
};
</script>