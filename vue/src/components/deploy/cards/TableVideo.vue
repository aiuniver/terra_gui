<template>
  <div>
    <Media :kind="'video'" :controls="true" :src="src" :style="{ width: '300px' }" :key="src"></Media>
  </div>
</template>

<script>
export default {
  components: {
    Media: () => import('@dongido/vue-viaudio'),
  },
  props: {
    value: { type: String, default: '' },
  },
  computed: {
    src() {
       return `/_media/blank/?path=${this.value}&r=${Date.now()}`;
    },
  },
};
</script>