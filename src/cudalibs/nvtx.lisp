(in-package petalisp-cuda.cudalibs)
;; next section imported from file /usr/local/include/nvtx3/nvToolsExt.h

(cffi:defctype wchar-t :uint32)

;(cffi:defctype int-fast16-t :long)

;(cffi:defctype int-fast32-t :long)

;(cffi:defctype int-fast64-t :long)

;(cffi:defctype uint-fast8-t :unsigned-char)

;(cffi:defctype uint-fast16-t :unsigned-long)

;(cffi:defctype uint-fast32-t :unsigned-long)

;(cffi:defctype uint-fast64-t :unsigned-long)

;(cffi:defctype intptr-t :long)

;(cffi:defctype uintptr-t :unsigned-long)

(cffi:defctype nvtxrangeid-t :uint64)

(cffi:defcstruct nvtxstringhandle)

(cffi:defctype nvtxstringhandle-t (:pointer (:struct nvtxStringHandle)))

(cffi:defcstruct nvtxdomainhandle)

(cffi:defctype nvtxdomainhandle-t (:pointer (:struct nvtxDomainHandle)))

(cffi:defcenum nvtxcolortype-t
  "---------------------------------------------------------------------------
Color Types
------------------------------------------------------------------------- */"
  (:nvtx-color-unknown 0)
  (:nvtx-color-argb 1))

(cffi:defctype nvtxcolortype-t :int ; enum nvtxColorType-t
)

(cffi:defcenum nvtxmessagetype-t
  "---------------------------------------------------------------------------
Message Types
------------------------------------------------------------------------- */"
  (:nvtx-message-unknown 0)
  (:nvtx-message-type-ascii 1)
  (:nvtx-message-type-unicode 2)
  (:nvtx-message-type-registered 3))

(cffi:defctype nvtxmessagetype-t :int ; enum nvtxMessageType-t
)

(cffi:defcunion nvtxmessagevalue-t
  (ascii (:pointer :char))
  (unicode (:pointer wchar-t))
  (registered nvtxStringHandle-t))

(cffi:defctype nvtxmessagevalue-t (:union nvtxMessageValue-t))

(cffi:defcenum nvtxinitializationmode-t
  "---------------------------------------------------------------------------
Initialization Modes
------------------------------------------------------------------------- */"
  (:nvtx-initialization-mode-unknown 0)
  (:nvtx-initialization-mode-callback-v1 1)
  (:nvtx-initialization-mode-callback-v2 2)
  (:nvtx-initialization-mode-size 3))

(cffi:defctype nvtxinitializationmode-t :int ; enum nvtxInitializationMode-t
)

(cffi:defcstruct nvtxinitializationattributes-v2
  "\brief Initialization Attribute Structure.
\anchor INITIALIZATION_ATTRIBUTE_STRUCTURE

This structure is used to describe the attributes used for initialization
of the NVTX API.

\par Initializing the Attributes

The caller should always perform the following three tasks when using
attributes:
<ul>
   <li>Zero the structure
   <li>Set the version field
   <li>Set the size field
</ul>

Zeroing the structure sets all the event attributes types and values
to the default value.

The version and size field are used by the Tools Extension
implementation to handle multiple versions of the attributes structure.
NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE may be used for the size.

It is recommended that the caller use one of the following to methods
to initialize the event attributes structure:

\par Method 1: Initializing nvtxInitializationAttributes_t for future compatibility
\code
nvtxInitializationAttributes_t initAttribs = {0};
initAttribs.version = NVTX_VERSION;
initAttribs.size = NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE;
\endcode

\par Method 2: Initializing nvtxInitializationAttributes_t for a specific version
\code
nvtxInitializationAttributes_t initAttribs = {0};
initAttribs.version =2;
initAttribs.size = (uint16_t)(sizeof(nvtxInitializationAttributes_v2));
\endcode

If the caller uses Method 1 it is critical that the entire binary
layout of the structure be configured to 0 so that all fields
are initialized to the default value.

The caller should either use both NVTX_VERSION and
NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
and a versioned type (Method 2).  Using a mix of the two methods
will likely cause either source level incompatibility or binary
incompatibility in the future.

\par Settings Attribute Types and Values


\par Example:
\code
// Initialize
nvtxInitializationAttributes_t initAttribs = {0};
initAttribs.version = NVTX_VERSION;
initAttribs.size = NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE;

// Configure the Attributes
initAttribs.mode = NVTX_INITIALIZATION_MODE_CALLBACK_V2;
initAttribs.fnptr = InitializeInjectionNvtx2;
\endcode
\sa
::nvtxInitializationMode_t
::nvtxInitialize"
  (version :uint16)
  (size :uint16)
  (mode :uint32)
  (fnptr :pointer #| function ptr void (void) |#))

(cffi:defctype nvtxinitializationattributes-v2 (:struct nvtxInitializationAttributes-v2))

(cffi:defctype nvtxinitializationattributes-t (:struct nvtxInitializationAttributes-v2))

(cffi:defcfun "nvtxinitialize" :int
  "\brief Force initialization (optional on most platforms)

Force NVTX library to initialize.  On some platform NVTX will implicit initialize
upon the first function call into an NVTX API.

\return Result codes are simplest to assume NVTX_SUCCESS or !NVTX_SUCCESS

\param initAttrib - The initialization attribute structure

\sa
::nvtxInitializationAttributes_t

\version \NVTX_VERSION_2
@{ */"
  (initattrib (:pointer nvtxInitializationAttributes-t)))

(cffi:defcenum nvtxpayloadtype-t
  "---------------------------------------------------------------------------
Payload Types
------------------------------------------------------------------------- */"
  (:nvtx-payload-unknown 0)
  (:nvtx-payload-type-unsigned-int64 1)
  (:nvtx-payload-type-int64 2)
  (:nvtx-payload-type-double 3)
  (:nvtx-payload-type-unsigned-int32 4)
  (:nvtx-payload-type-int32 5)
  (:nvtx-payload-type-float 6))

(cffi:defctype nvtxpayloadtype-t :int ; enum nvtxPayloadType-t
)

    ;union payload_t
    ;{
        ;uint64_t ullValue;
        ;int64_t llValue;
        ;double dValue;
        ;/* NVTX_VERSION_2 */
        ;uint32_t uiValue;
        ;int32_t iValue;
        ;float fValue;
    ;} payload;

;(defcunion payload-t
  ;(int-value :unsigned-int)
  ;(double-value :double))

(cffi:defcstruct nvtxeventattributes-v2
  "\brief Event Attribute Structure.
\anchor EVENT_ATTRIBUTE_STRUCTURE

This structure is used to describe the attributes of an event. The layout of
the structure is defined by a specific version of the tools extension
library and can change between different versions of the Tools Extension
library.

\par Initializing the Attributes

The caller should always perform the following three tasks when using
attributes:
<ul>
   <li>Zero the structure
   <li>Set the version field
   <li>Set the size field
</ul>

Zeroing the structure sets all the event attributes types and values
to the default value.

The version and size field are used by the Tools Extension
implementation to handle multiple versions of the attributes structure.

It is recommended that the caller use one of the following to methods
to initialize the event attributes structure:

\par Method 1: Initializing nvtxEventAttributes for future compatibility
\code
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
\endcode

\par Method 2: Initializing nvtxEventAttributes for a specific version
\code
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = 1;
eventAttrib.size = (uint16_t)(sizeof(nvtxEventAttributes_v1));
\endcode

If the caller uses Method 1 it is critical that the entire binary
layout of the structure be configured to 0 so that all fields
are initialized to the default value.

The caller should either use both NVTX_VERSION and
NVTX_EVENT_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
and a versioned type (Method 2).  Using a mix of the two methods
will likely cause either source level incompatibility or binary
incompatibility in the future.

\par Settings Attribute Types and Values


\par Example:
\code
// Initialize
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

// Configure the Attributes
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.color = 0xFF880000;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.message.ascii = \"Example\";
\endcode

In the example the caller does not have to set the value of
\ref ::nvtxEventAttributes_v2::category or
\ref ::nvtxEventAttributes_v2::payload as these fields were set to
the default value by {0}.
\sa
::nvtxDomainMarkEx
::nvtxDomainRangeStartEx
::nvtxDomainRangePushEx"
  (version :uint16)
  (size :uint16)
  (category :uint32)
  (colortype :int32)
  (color :uint32)
  (payloadtype :int32)
  (reserved0 :int32)
  ( payload :int64)
  (messagetype :int32)
  (message nvtxMessageValue-t))

(cffi:defctype nvtxeventattributes-v2 (:struct nvtxEventAttributes-v2))

(cffi:defctype nvtxeventattributes-t (:struct nvtxEventAttributes-v2))

(cffi:defcfun "nvtxdomainmarkex" :void
  "\brief Marks an instantaneous event in the application.

A marker can contain a text message or specify additional information
using the event attributes structure.  These attributes include a text
message, color, category, and a payload. Each of the attributes is optional
and can only be sent out using the \ref nvtxDomainMarkEx function.

nvtxDomainMarkEx(NULL, event) is equivalent to calling
nvtxMarkEx(event).

\param domain    - The domain of scoping the category.
\param eventAttrib - The event attribute structure defining the marker's
attribute types and attribute values.

\sa
::nvtxMarkEx

\version \NVTX_VERSION_2
@{ */"
  (domain nvtxDomainHandle-t)
  (eventattrib (:pointer nvtxEventAttributes-t)))

(cffi:defcfun "nvtxmarkex" :void
  "\brief Marks an instantaneous event in the application.

A marker can contain a text message or specify additional information
using the event attributes structure.  These attributes include a text
message, color, category, and a payload. Each of the attributes is optional
and can only be sent out using the \ref nvtxMarkEx function.
If \ref nvtxMarkA or \ref nvtxMarkW are used to specify the marker
or if an attribute is unspecified then a default value will be used.

\param eventAttrib - The event attribute structure defining the marker's
attribute types and attribute values.

\par Example:
\code
// zero the structure
nvtxEventAttributes_t eventAttrib = {0};
// set the version and the size information
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
// configure the attributes.  0 is the default for all attributes.
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.color = 0xFF880000;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.message.ascii = \"Example nvtxMarkEx\";
nvtxMarkEx(&eventAttrib);
\endcode

\sa
::nvtxDomainMarkEx

\version \NVTX_VERSION_1
@{ */"
  (eventattrib (:pointer nvtxEventAttributes-t)))

(cffi:defcfun "nvtxmarka" :void
  "\brief Marks an instantaneous event in the application.

A marker created using \ref nvtxMarkA or \ref nvtxMarkW contains only a
text message.

\param message     - The message associated to this marker event.

\par Example:
\code
nvtxMarkA(\"Example nvtxMarkA\");
nvtxMarkW(L\"Example nvtxMarkW\");
\endcode

\sa
::nvtxDomainMarkEx
::nvtxMarkEx

\version \NVTX_VERSION_0
@{ */"
  (message (:pointer :char)))

(cffi:defcfun "nvtxmarkw" :void
  (message (:pointer wchar-t)))

(cffi:defcfun "nvtxdomainrangestartex" nvtxRangeId-t
  "\brief Starts a process range in a domain.

\param domain    - The domain of scoping the category.
\param eventAttrib - The event attribute structure defining the range's
attribute types and attribute values.

\return The unique ID used to correlate a pair of Start and End events.

\remarks Ranges defined by Start/End can overlap.

\par Example:
\code
nvtxDomainHandle_t domain = nvtxDomainCreateA(\"my domain\");
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.message.ascii = \"my range\";
nvtxRangeId_t rangeId = nvtxDomainRangeStartEx(&eventAttrib);
// ...
nvtxDomainRangeEnd(rangeId);
\endcode

\sa
::nvtxDomainRangeEnd

\version \NVTX_VERSION_2
@{ */"
  (domain nvtxDomainHandle-t)
  (eventattrib (:pointer nvtxEventAttributes-t)))

(cffi:defcfun "nvtxrangestartex" nvtxRangeId-t
  "\brief Starts a process range.

\param eventAttrib - The event attribute structure defining the range's
attribute types and attribute values.

\return The unique ID used to correlate a pair of Start and End events.

\remarks Ranges defined by Start/End can overlap.

\par Example:
\code
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.category = 3;
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.color = 0xFF0088FF;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.message.ascii = \"Example Range\";
nvtxRangeId_t rangeId = nvtxRangeStartEx(&eventAttrib);
// ...
nvtxRangeEnd(rangeId);
\endcode

\sa
::nvtxRangeEnd
::nvtxDomainRangeStartEx

\version \NVTX_VERSION_1
@{ */"
  (eventattrib (:pointer nvtxEventAttributes-t)))

(cffi:defcfun "nvtxrangestarta" nvtxRangeId-t
  "\brief Starts a process range.

\param message     - The event message associated to this range event.

\return The unique ID used to correlate a pair of Start and End events.

\remarks Ranges defined by Start/End can overlap.

\par Example:
\code
nvtxRangeId_t r1 = nvtxRangeStartA(\"Range 1\");
nvtxRangeId_t r2 = nvtxRangeStartW(L\"Range 2\");
nvtxRangeEnd(r1);
nvtxRangeEnd(r2);
\endcode

\sa
::nvtxRangeEnd
::nvtxRangeStartEx
::nvtxDomainRangeStartEx

\version \NVTX_VERSION_0
@{ */"
  (message (:pointer :char)))

(cffi:defcfun "nvtxrangestartw" nvtxRangeId-t
  (message (:pointer wchar-t)))

(cffi:defcfun "nvtxdomainrangeend" :void
  "\brief Ends a process range.

\param domain - The domain 
\param id - The correlation ID returned from a nvtxRangeStart call.

\remarks This function is offered completeness but is an alias for ::nvtxRangeEnd. 
It does not need a domain param since that is associated iwth the range ID at ::nvtxDomainRangeStartEx

\par Example:
\code
nvtxDomainHandle_t domain = nvtxDomainCreateA(\"my domain\");
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.message.ascii = \"my range\";
nvtxRangeId_t rangeId = nvtxDomainRangeStartEx(&eventAttrib);
// ...
nvtxDomainRangeEnd(rangeId);
\endcode

\sa
::nvtxDomainRangeStartEx

\version \NVTX_VERSION_2
@{ */"
  (domain nvtxDomainHandle-t)
  (id nvtxRangeId-t))

(cffi:defcfun "nvtxrangeend" :void
  "\brief Ends a process range.

\param id - The correlation ID returned from an nvtxRangeStart call.

\sa
::nvtxDomainRangeStartEx
::nvtxRangeStartEx
::nvtxRangeStartA
::nvtxRangeStartW

\version \NVTX_VERSION_0
@{ */"
  (id nvtxRangeId-t))

(cffi:defcfun "nvtxdomainrangepushex" :int
  "\brief Starts a nested thread range.

\param domain    - The domain of scoping.
\param eventAttrib - The event attribute structure defining the range's
attribute types and attribute values.

\return The 0 based level of range being started. This value is scoped to the domain.
If an error occurs, a negative value is returned.

\par Example:
\code
nvtxDomainHandle_t domain = nvtxDomainCreateA(\"example domain\");
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.color = 0xFFFF0000;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.message.ascii = \"Level 0\";
nvtxDomainRangePushEx(domain, &eventAttrib);

// Re-use eventAttrib
eventAttrib.messageType = NVTX_MESSAGE_TYPE_UNICODE;
eventAttrib.message.unicode = L\"Level 1\";
nvtxDomainRangePushEx(domain, &eventAttrib);

nvtxDomainRangePop(domain); //level 1
nvtxDomainRangePop(domain); //level 0
\endcode

\sa
::nvtxDomainRangePop

\version \NVTX_VERSION_2
@{ */"
  (domain nvtxDomainHandle-t)
  (eventattrib (:pointer nvtxEventAttributes-t)))

(cffi:defcfun "nvtxrangepushex" :int
  "\brief Starts a nested thread range.

\param eventAttrib - The event attribute structure defining the range's
attribute types and attribute values.

\return The 0 based level of range being started. This level is per domain.
If an error occurs a negative value is returned.

\par Example:
\code
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.color = 0xFFFF0000;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.message.ascii = \"Level 0\";
nvtxRangePushEx(&eventAttrib);

// Re-use eventAttrib
eventAttrib.messageType = NVTX_MESSAGE_TYPE_UNICODE;
eventAttrib.message.unicode = L\"Level 1\";
nvtxRangePushEx(&eventAttrib);

nvtxRangePop();
nvtxRangePop();
\endcode

\sa
::nvtxDomainRangePushEx
::nvtxRangePop

\version \NVTX_VERSION_1
@{ */"
  (eventattrib (:pointer nvtxEventAttributes-t)))

(cffi:defcfun "nvtxRangePushA" :int
  "\brief Starts a nested thread range.

\param message     - The event message associated to this range event.

\return The 0 based level of range being started.  If an error occurs a
negative value is returned.

\par Example:
\code
nvtxRangePushA(\"Level 0\");
nvtxRangePushW(L\"Level 1\");
nvtxRangePop();
nvtxRangePop();
\endcode

\sa
::nvtxDomainRangePushEx
::nvtxRangePop

\version \NVTX_VERSION_0
@{ */"
  (message (:pointer :char)))

(cffi:defcfun "nvtxrangepushw" :int
  (message (:pointer wchar-t)))

(cffi:defcfun "nvtxdomainrangepop" :int
  "\brief Ends a nested thread range.

\return The level of the range being ended. If an error occurs a negative
value is returned on the current thread.

\par Example:
\code
nvtxDomainHandle_t domain = nvtxDomainCreate(\"example library\");
nvtxDomainRangePushA(domain, \"Level 0\");
nvtxDomainRangePushW(domain, L\"Level 1\");
nvtxDomainRangePop(domain);
nvtxDomainRangePop(domain);
\endcode

\sa
::nvtxRangePushEx
::nvtxRangePushA
::nvtxRangePushW

\version \NVTX_VERSION_2
@{ */"
  (domain nvtxDomainHandle-t))

(cffi:defcfun "nvtxRangePop" :int
  "\brief Ends a nested thread range.

\return The level of the range being ended. If an error occurs a negative
value is returned on the current thread.

\par Example:
\code
nvtxRangePushA(\"Level 0\");
nvtxRangePushW(L\"Level 1\");
nvtxRangePop();
nvtxRangePop();
\endcode

\sa
::nvtxRangePushEx
::nvtxRangePushA
::nvtxRangePushW

\version \NVTX_VERSION_0
@{ */")

(cffi:defcenum nvtxresourcegenerictype-t
  "\brief Generic resource type for when a resource class is not available.

\sa
::nvtxDomainResourceCreate

\version \NVTX_VERSION_2"
  (:nvtx-resource-type-unknown 0)
  (:nvtx-resource-type-generic-pointer 65537)
  (:nvtx-resource-type-generic-handle 65538)
  (:nvtx-resource-type-generic-thread-native 65539)
  (:nvtx-resource-type-generic-thread-posix 65540))

(cffi:defctype nvtxresourcegenerictype-t :int ; enum nvtxResourceGenericType-t
)

(cffi:defcstruct nvtxresourceattributes-v0
  "\brief Resource Attribute Structure.
\anchor RESOURCE_ATTRIBUTE_STRUCTURE

This structure is used to describe the attributes of a resource. The layout of
the structure is defined by a specific version of the tools extension
library and can change between different versions of the Tools Extension
library.

\par Initializing the Attributes

The caller should always perform the following three tasks when using
attributes:
<ul>
   <li>Zero the structure
   <li>Set the version field
   <li>Set the size field
</ul>

Zeroing the structure sets all the resource attributes types and values
to the default value.

The version and size field are used by the Tools Extension
implementation to handle multiple versions of the attributes structure.

It is recommended that the caller use one of the following to methods
to initialize the event attributes structure:

\par Method 1: Initializing nvtxEventAttributes for future compatibility
\code
nvtxResourceAttributes_t attribs = {0};
attribs.version = NVTX_VERSION;
attribs.size = NVTX_RESOURCE_ATTRIB_STRUCT_SIZE;
\endcode

\par Method 2: Initializing nvtxEventAttributes for a specific version
\code
nvtxResourceAttributes_v0 attribs = {0};
attribs.version = 2;
attribs.size = (uint16_t)(sizeof(nvtxResourceAttributes_v0));
\endcode

If the caller uses Method 1 it is critical that the entire binary
layout of the structure be configured to 0 so that all fields
are initialized to the default value.

The caller should either use both NVTX_VERSION and
NVTX_RESOURCE_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
and a versioned type (Method 2).  Using a mix of the two methods
will likely cause either source level incompatibility or binary
incompatibility in the future.

\par Settings Attribute Types and Values


\par Example:
\code
nvtxDomainHandle_t domain = nvtxDomainCreateA(\"example domain\");

// Initialize
nvtxResourceAttributes_t attribs = {0};
attribs.version = NVTX_VERSION;
attribs.size = NVTX_RESOURCE_ATTRIB_STRUCT_SIZE;

// Configure the Attributes
attribs.identifierType = NVTX_RESOURCE_TYPE_GENERIC_POINTER;
attribs.identifier.pValue = (const void*)pMutex;
attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
attribs.message.ascii = \"Single thread access to database.\";

nvtxResourceHandle_t handle = nvtxDomainResourceCreate(domain, attribs);
\endcode

\sa
::nvtxDomainResourceCreate"
  (version :uint16)
  (size :uint16)
  (identifiertype :int32)
  (identifier :int64)
  (messagetype :int32)
  (message nvtxMessageValue-t))

(cffi:defctype nvtxresourceattributes-v0 (:struct nvtxResourceAttributes-v0))

(cffi:defctype nvtxresourceattributes-t (:struct nvtxResourceAttributes-v0))

(cffi:defcstruct nvtxresourcehandle)

(cffi:defctype nvtxresourcehandle-t (:pointer (:struct nvtxResourceHandle)))

(cffi:defcfun "nvtxdomainresourcecreate" nvtxResourceHandle-t
  "\brief Create a resource object to track and associate data with OS and middleware objects

Allows users to associate an API handle or pointer with a user-provided name.


\param domain - Domain to own the resource object
\param attribs - Attributes to be associated with the resource

\return A handle that represents the newly created resource object.

\par Example:
\code
nvtxDomainHandle_t domain = nvtxDomainCreateA(\"example domain\");
nvtxResourceAttributes_t attribs = {0};
attribs.version = NVTX_VERSION;
attribs.size = NVTX_RESOURCE_ATTRIB_STRUCT_SIZE;
attribs.identifierType = NVTX_RESOURCE_TYPE_GENERIC_POINTER;
attribs.identifier.pValue = (const void*)pMutex;
attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
attribs.message.ascii = \"Single thread access to database.\";
nvtxResourceHandle_t handle = nvtxDomainResourceCreate(domain, attribs);
\endcode

\sa
::nvtxResourceAttributes_t
::nvtxDomainResourceDestroy

\version \NVTX_VERSION_2
@{ */"
  (domain nvtxDomainHandle-t)
  (attribs (:pointer nvtxResourceAttributes-t)))

(cffi:defcfun "nvtxdomainresourcedestroy" :void
  "\brief Destroy a resource object to track and associate data with OS and middleware objects

Allows users to associate an API handle or pointer with a user-provided name.

\param resource - Handle to the resource in which to operate.

\par Example:
\code
nvtxDomainHandle_t domain = nvtxDomainCreateA(\"example domain\");
nvtxResourceAttributes_t attribs = {0};
attribs.version = NVTX_VERSION;
attribs.size = NVTX_RESOURCE_ATTRIB_STRUCT_SIZE;
attribs.identifierType = NVTX_RESOURCE_TYPE_GENERIC_POINTER;
attribs.identifier.pValue = (const void*)pMutex;
attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
attribs.message.ascii = \"Single thread access to database.\";
nvtxResourceHandle_t handle = nvtxDomainResourceCreate(domain, attribs);
nvtxDomainResourceDestroy(handle);
\endcode

\sa
::nvtxDomainResourceCreate

\version \NVTX_VERSION_2
@{ */"
  (resource nvtxResourceHandle-t))

(cffi:defcfun "nvtxdomainnamecategorya" :void
  "\brief Annotate an NVTX category used within a domain.

Categories are used to group sets of events. Each category is identified
through a unique ID and that ID is passed into any of the marker/range
events to assign that event to a specific category. The nvtxDomainNameCategory
function calls allow the user to assign a name to a category ID that is
specific to the domain.

nvtxDomainNameCategory(NULL, category, name) is equivalent to calling
nvtxNameCategory(category, name).

\param domain    - The domain of scoping the category.
\param category  - The category ID to name.
\param name      - The name of the category.

\remarks The category names are tracked per domain.

\par Example:
\code
nvtxDomainHandle_t domain = nvtxDomainCreateA(\"example\");
nvtxDomainNameCategoryA(domain, 1, \"Memory Allocation\");
nvtxDomainNameCategoryW(domain, 2, L\"Memory Transfer\");
\endcode

\version \NVTX_VERSION_2
@{ */"
  (domain nvtxDomainHandle-t)
  (category :uint32)
  (name (:pointer :char)))

(cffi:defcfun "nvtxdomainnamecategoryw" :void
  (domain nvtxDomainHandle-t)
  (category :uint32)
  (name (:pointer wchar-t)))

(cffi:defcfun "nvtxnamecategorya" :void
  "\brief Annotate an NVTX category.

Categories are used to group sets of events. Each category is identified
through a unique ID and that ID is passed into any of the marker/range
events to assign that event to a specific category. The nvtxNameCategory
function calls allow the user to assign a name to a category ID.

\param category - The category ID to name.
\param name     - The name of the category.

\remarks The category names are tracked per process.

\par Example:
\code
nvtxNameCategory(1, \"Memory Allocation\");
nvtxNameCategory(2, \"Memory Transfer\");
nvtxNameCategory(3, \"Memory Object Lifetime\");
\endcode

\version \NVTX_VERSION_1
@{ */"
  (category :uint32)
  (name (:pointer :char)))

(cffi:defcfun "nvtxnamecategoryw" :void
  (category :uint32)
  (name (:pointer wchar-t)))

(cffi:defcfun "nvtxnameosthreada" :void
  "\brief Annotate an OS thread.

Allows the user to name an active thread of the current process. If an
invalid thread ID is provided or a thread ID from a different process is
used the behavior of the tool is implementation dependent.

The thread name is associated to the default domain.  To support domains 
use resource objects via ::nvtxDomainResourceCreate.

\param threadId - The ID of the thread to name.
\param name     - The name of the thread.

\par Example:
\code
nvtxNameOsThread(GetCurrentThreadId(), \"MAIN_THREAD\");
\endcode

\version \NVTX_VERSION_1
@{ */"
  (threadid :uint32)
  (name (:pointer :char)))

(cffi:defcfun "nvtxnameosthreadw" :void
  (threadid :uint32)
  (name (:pointer wchar-t)))

(cffi:defcfun "nvtxdomainregisterstringa" nvtxStringHandle-t
  "\brief Register a string.
Registers an immutable string with NVTX. Once registered the pointer used
to register the domain name can be used in nvtxEventAttributes_t
\ref MESSAGE_FIELD. This allows NVTX implementation to skip copying the
contents of the message on each event invocation.

String registration is an optimization. It is recommended to use string
registration if the string will be passed to an event many times.

String are not unregistered, except that by unregistering the entire domain

\param domain  - Domain handle. If NULL then the global domain is used.
\param string    - A unique pointer to a sequence of characters.

\return A handle representing the registered string.

\par Example:
\code
nvtxDomainCreateA(\"com.nvidia.nvtx.example\");
nvtxStringHandle_t message = nvtxDomainRegisterStringA(domain, \"registered string\");
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
eventAttrib.message.registered = message;
\endcode

\version \NVTX_VERSION_2
@{ */"
  (domain nvtxDomainHandle-t)
  (string (:pointer :char)))

(cffi:defcfun "nvtxdomainregisterstringw" nvtxStringHandle-t
  (domain nvtxDomainHandle-t)
  (string (:pointer wchar-t)))

(cffi:defcfun "nvtxdomaincreatea" nvtxDomainHandle-t
  "\brief Register a NVTX domain.

Domains are used to scope annotations. All NVTX_VERSION_0 and NVTX_VERSION_1
annotations are scoped to the global domain. The function nvtxDomainCreate
creates a new named domain.

Each domain maintains its own nvtxRangePush and nvtxRangePop stack.

\param name - A unique string representing the domain.

\return A handle representing the domain.

\par Example:
\code
nvtxDomainHandle_t domain = nvtxDomainCreateA(\"com.nvidia.nvtx.example\");

nvtxMarkA(\"nvtxMarkA to global domain\");

nvtxEventAttributes_t eventAttrib1 = {0};
eventAttrib1.version = NVTX_VERSION;
eventAttrib1.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib1.message.ascii = \"nvtxDomainMarkEx to global domain\";
nvtxDomainMarkEx(NULL, &eventAttrib1);

nvtxEventAttributes_t eventAttrib2 = {0};
eventAttrib2.version = NVTX_VERSION;
eventAttrib2.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib2.message.ascii = \"nvtxDomainMarkEx to com.nvidia.nvtx.example\";
nvtxDomainMarkEx(domain, &eventAttrib2);
nvtxDomainDestroy(domain);
\endcode

\sa
::nvtxDomainDestroy

\version \NVTX_VERSION_2
@{ */"
  (name (:pointer :char)))

(cffi:defcfun "nvtxdomaincreatew" nvtxDomainHandle-t
  (name (:pointer wchar-t)))

(cffi:defcfun "nvtxdomaindestroy" :void
  "\brief Unregister a NVTX domain.

Unregisters the domain handle and frees all domain specific resources.

\param domain    - the domain handle

\par Example:
\code
nvtxDomainHandle_t domain = nvtxDomainCreateA(\"com.nvidia.nvtx.example\");
nvtxDomainDestroy(domain);
\endcode

\sa
::nvtxDomainCreateA
::nvtxDomainCreateW

\version \NVTX_VERSION_2
@{ */"
  (domain nvtxDomainHandle-t))

(cl:defmacro with-nvtx-range (name cl:&body body)  
  (cl:if *nvtx-found*
      `(cffi:with-foreign-string (name-ptr ,name)
         (nvtxRangePushA name-ptr)
         (cl:unwind-protect
             (cl:locally ,@body)
           (nvtxRangePop)))
      `(cl:progn
         ,@body)))

(cl:defun nvtxMark (name)
  (cffi:with-foreign-string (name-ptr name)
    (nvtxMarkA name-ptr)))
