<template>
  <div>
    <TableImage v-if="type === 'image'" v-bind="data" />
    <TableText v-if="type === 'str' || type === 'number'" v-bind="data" />
    <TableTag v-if="type === 'text'" :data="data" />
    <TableVideo v-if="type === 'video'" :src="data" />
    <TableAudio v-if="type === 'audio'" :url="data" />
  </div>
</template>

<script>
export default {
  name: 't-forms-prediction',
  components: {
    TableImage: () => import('./TableImage'),
    TableAudio: () => import('./TableAudio'),
    TableText: () => import('./TableText'),
    TableTag: () => import('./TableTag'),
    TableVideo: () => import('./TableVideo.vue'),
  },
  props: {
    type: {
      type: String,
      default: ''
    },
    data: {
      type: Object,
      default: () => {}
    },
  },
};
</script>

<style lang="scss" scoped>
</style>