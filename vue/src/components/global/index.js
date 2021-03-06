import FilesMenu from './blocks/FilesMenu';
import TAutoField from './blocks/TAutoField';
import TAutoFieldHeandler from './blocks/TAutoFieldHeandler';
import TAutoFieldTrainings from './blocks/TAutoFieldTrainings';
import TAutoFieldDeploy from './blocks/TAutoFieldDeploy';
import TAutoFieldCascade from './blocks/TAutoFieldCascade';

import Button from './forms/Button';
import Autocomplete from './forms/Autocomplete';
import SegmentationManual from './forms/SegmentationManual';
import SegmentationSearch from './forms/SegmentationSearch';
import SegmentationAnnotation from './forms/SegmentationAnnotation';
import MultiSelect from './forms/MultiSelect';
import Checkbox from './forms/Checkbox';
import Input from './forms/Input';
import Select from './forms/Select';
import SelectTasks from './forms/SelectTasks';
import TupleCascade from './forms/TupleCascade';


// _______________NEW____________________//
import TInput from './new/forms/TInput';
import TCheckbox from './new/forms/TCheckbox';
import TSelect from './new/forms/TSelect';
import TField from './forms/TField';
import TAutoComplete from './new/forms/TAutoComplete';
import TAutoCompleteTwo from './new/forms/TAutoCompleteTwo';
import DDropdown from './forms/DDropdown';





const components = [
  FilesMenu,
  TAutoField,
  TAutoFieldHeandler,
  TAutoFieldTrainings,
  TAutoFieldDeploy,
  TAutoFieldCascade,
  Button,
  Checkbox,
  Input,
  Select,
  SegmentationManual,
  SegmentationSearch,
  SegmentationAnnotation,
  Autocomplete,
  MultiSelect,
  SelectTasks,
  TupleCascade,

  TInput,
  TCheckbox,
  TSelect,
  TAutoComplete,
  TAutoCompleteTwo,
  TField,
  DDropdown,
];

import Vue from 'vue';
const requireComponent = require.context('@/components/global', true, /\.vue$/);

components.forEach(component => Vue.component(component.name, component))

requireComponent.keys().forEach(fileName => {
  const componentConfig = requireComponent(fileName);
  const componentName = fileName.replace(/^.*[\\/]/, '').replace(/\.\w+$/, '');
  Vue.component(componentName, componentConfig.default || componentConfig);
});

