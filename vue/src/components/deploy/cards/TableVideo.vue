<template>
  <div>
    <Media :kind="'video'" :controls="true" :src="src" :style="{ width: '300px' }" :key="src"></Media>
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
      default: '2323',
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