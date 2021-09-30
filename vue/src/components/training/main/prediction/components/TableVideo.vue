<template>
  <div class="t-predict-video">
    <!-- <iframe 
      loading="lazy"
      sandbox="allow-forms allow-scripts allow-pointer-lock allow-same-origin allow-top-navigation allow-presentation"
      class="t-predict-video__item"
      allowfullscreen
      :src="src"
    ></iframe> -->
    <video-embed :src="src"></video-embed>
  </div>
</template>



<script>
export default {
  props: {
    value: { type: String, default: '' },
  },
  computed: {
    src() {
      return `/_media/blank/?path=${this.value}&r=${this.update}`
    }
  },
  data:() => ({
    autoStart: false
  }),
};
</script>