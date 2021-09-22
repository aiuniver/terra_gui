<template>
  <div>
    <TableImage v-if="type === 'image'" v-bind="data" />
    <TableText v-if="type === 'str' || type === 'number'" v-bind="data" />
    <TableTag v-if="type === 'text'" :data="data" />
    <Embed v-if="type === 'video'" :src="data"></Embed>
    <TableAudio v-if="type === 'audio'" :url="data" />
  </div>
</template>

<script>
import TableImage from '../components/TableImage';
import TableText from '../components/TableText';
// import TableStatisticText from './components/TableStatisticText';
import TableTag from './TableTag.vue';
import TableAudio from '../components/TableAudio';
import Embed from 'v-video-embed/src/embed';
export default {
  name: 't-forms-prediction',
  components: {
    TableImage,
    // TableStatisticText,
    TableAudio,
    TableText,
    TableTag,
    Embed,
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